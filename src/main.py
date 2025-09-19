from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
import discord
from discord.ext import commands
import yaml

from .ark_agent import ARKAgent
from .discord_voice import VoicePipeline
from .router import Router
from .tts_realtime import TTSClient
from .commands_cog import ARKCommands
from .memory import create_memory_store, MemoryStore


def setup_logging() -> None:
    level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    fh = logging.FileHandler("arkbot.log", encoding="utf-8")
    fh.setLevel(getattr(logging, level, logging.INFO))
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logging.getLogger().addHandler(fh)


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def ensure_opus_loaded() -> None:
    try:
        if discord.opus.is_loaded():
            return
    except Exception:
        # Older py-cord may not expose is_loaded; attempt load regardless
        pass
    candidates = [
        os.environ.get("OPUS_LIBRARY_PATH"),
        "libopus-0.dll",
        "libopus.so.0",
        "libopus.so",
        "libopus.dylib",
        "/opt/homebrew/opt/opus/lib/libopus.0.dylib",
    ]
    for cand in candidates:
        if not cand:
            continue
        try:
            discord.opus.load_opus(cand)
            if discord.opus.is_loaded():
                logging.getLogger(__name__).info("Loaded opus library: %s", cand)
                return
        except Exception:
            continue
    logging.getLogger(__name__).warning("Opus library not explicitly loaded; if voice receive fails, set OPUS_LIBRARY_PATH.")


class ARKBot(commands.Bot):
    def __init__(self) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        intents.members = True
        app_id = os.environ.get("DISCORD_APP_ID")
        if app_id and str(app_id).isdigit():
            super().__init__(command_prefix="!", intents=intents, application_id=int(app_id))
        else:
            super().__init__(command_prefix="!", intents=intents)

        self.voice_pipeline: Optional[VoicePipeline] = None
        self.router: Optional[Router] = None
        self.ark_agent: Optional[ARKAgent] = None
        self.tts: Optional[TTSClient] = None
        self.memory_store: Optional[MemoryStore] = None
        # Ensure guild id is available early for on_ready
        self._guild_id_env = os.environ.get("GUILD_ID")

    async def setup_hook(self) -> None:
        # Load configs
        base = Path(__file__).resolve().parent
        campaign = load_yaml(base / "config" / "campaign.yml")
        self.campaign = campaign
        mapping = load_yaml(base / "config" / "mapping.yml")
        # Prefer env CONTEXT_TURNS, then campaign.ark.context_window, else default 12
        context_turns_env = os.environ.get("CONTEXT_TURNS")
        if context_turns_env is not None and str(context_turns_env).strip():
            context_turns = int(context_turns_env)
        else:
            context_turns = int(campaign.get("ark", {}).get("context_window", 12))
        self.router = Router(campaign, mapping, context_max_turns=context_turns)
        asr_cfg = campaign.get("asr", {}) or {}
        log_enabled = bool(asr_cfg.get("log_enabled", False))
        backend = os.environ.get("ARK_LOG_BACKEND") or asr_cfg.get("log_backend") or "json"
        log_file = os.environ.get("ARK_LOG_FILE") or asr_cfg.get("log_file") or "data/session_log.json"
        if log_enabled:
            try:
                self.memory_store = create_memory_store(backend, log_file)
                logging.getLogger(__name__).info("Memory store ready (backend=%s, file=%s)", backend, log_file)
            except Exception:
                logging.getLogger(__name__).exception("Failed to initialize memory store")
                self.memory_store = None
        else:
            self.memory_store = None
        if self.router:
            logging.getLogger(__name__).info(
                "Wake detection initialized (enabled=%s, primary=%s, fallback=%s, threshold=%.2f)",
                self.router.asr_kws_enabled,
                self.router.wake_word,
                self.router.asr_kws_wake_phrase,
                self.router.asr_confidence_threshold,
            )
        system_prompt = (
            campaign.get("ark", {}).get("system_prompt")
            or campaign.get("system_prompt", "")
        )
        self.ark_agent = ARKAgent(api_key=os.environ.get("OPENAI_API_KEY", ""), system_prompt=system_prompt)
        # TTS config: prefer env model, else campaign tts.model; voice from campaign if present
        tts_model = os.environ.get("OPENAI_TTS_MODEL") or campaign.get("tts", {}).get("model")
        tts_voice = campaign.get("tts", {}).get("voice") or "alloy"
        if tts_model:
            self.tts = TTSClient(api_key=os.environ.get("OPENAI_API_KEY", ""), model=tts_model, voice=tts_voice)
        # Apply env overrides for wake-word and enable state if provided
        ww = os.environ.get("ARK_WAKE_WORD")
        if ww and self.router:
            self.router.set_wake_word(ww)
        enabled = os.environ.get("ARK_ENABLED")
        if enabled is not None and self.router:
            val = str(enabled).lower() in ("1", "true", "yes", "on")
            self.router.set_ark_enabled(val)

        @self.command(name="join")
        async def join(ctx: commands.Context):
            if not ctx.author.voice or not ctx.author.voice.channel:
                return await ctx.reply("Сначала подключись к голосовому каналу.")
            channel = ctx.author.voice.channel
            voice = discord.utils.get(self.voice_clients, guild=ctx.guild)
            if voice and voice.is_connected():
                await voice.move_to(channel)
            else:
                voice = await channel.connect()
            await ctx.reply(f"Подключился к {channel.name}. Начинаю приём аудио.")

            # Start voice pipeline
            assert self.router and self.ark_agent
            # ASR models: realtime from env, fallback from env or campaign.asr.model
            rt_model = os.environ.get("OPENAI_REALTIME_MODEL", "gpt-realtime")
            fb_model = os.environ.get("OPENAI_WHISPER_MODEL") or campaign.get("asr", {}).get("model", "whisper-1")

            self.voice_pipeline = VoicePipeline(
                voice=voice,
                router=self.router,
                ark=self.ark_agent,
                tts=self.tts,
                openai_key=os.environ.get("OPENAI_API_KEY", ""),
                rt_model=rt_model,
                whisper_model=fb_model,
                memory=self.memory_store,
            )
            await self.voice_pipeline.start()

        @self.command(name="leave")
        async def leave(ctx: commands.Context):
            voice = discord.utils.get(self.voice_clients, guild=ctx.guild)
            if voice and voice.is_connected():
                if self.voice_pipeline:
                    self.voice_pipeline.stop()
                    self.voice_pipeline = None
                await voice.disconnect()
                await ctx.reply("Отключился от голосового канала.")
            else:
                await ctx.reply("Я не в голосовом канале.")

        @self.command(name="set-character")
        async def set_character(ctx: commands.Context, member: discord.Member, *, character: str):
            assert self.router
            self.router.set_character(member.id, character)
            await ctx.reply(f"Назначен персонаж для {member.mention}: {character}")

        @self.group(name="ark", invoke_without_command=True)
        async def ark_group(ctx: commands.Context):
            await ctx.reply("Используй: !ark on | !ark off | !ark wakeword <слово> | !ark test | !ark recall <запрос> | !ark clear-log")

        @ark_group.command(name="on")
        async def ark_on(ctx: commands.Context):
            assert self.router
            self.router.set_ark_enabled(True)
            await ctx.reply("ARK активирован.")

        @ark_group.command(name="off")
        async def ark_off(ctx: commands.Context):
            assert self.router
            self.router.set_ark_enabled(False)
            await ctx.reply("ARK отключен.")

        @ark_group.command(name="wakeword")
        async def ark_wakeword(ctx: commands.Context, *, word: str):
            assert self.router
            self.router.set_wake_word(word)
            await ctx.reply(f"Wake-word установлен: {word}")

        @ark_group.command(name="test")
        async def ark_test(ctx: commands.Context, *, text: str = "Проверка связи."):
            assert self.router and self.ark_agent
            reply = await self.ark_agent.reply(self.router.context(), text, None)
            if self.tts:
                try:
                    path = await self.tts.speak_to_tempfile(reply)
                    voice = discord.utils.get(self.voice_clients, guild=ctx.guild)
                    if not voice or not voice.is_connected():
                        await ctx.reply(reply)
                    else:
                        # Play via voice
                        from .discord_voice import VoicePipeline as _VP

                        vp = _VP(
                            voice,
                            self.router,
                            self.ark_agent,
                            self.tts,
                            os.environ.get("OPENAI_API_KEY", ""),
                            os.environ.get("OPENAI_REALTIME_MODEL", "gpt-4o-realtime-preview"),
                            os.environ.get("OPENAI_WHISPER_MODEL", "whisper-1"),
                            memory=self.memory_store,
                        )
                        await vp._play_file(path)
                except Exception:
                    await ctx.reply(reply)
            else:
                await ctx.reply(reply)

        @ark_group.command(name="recall")
        async def ark_recall(ctx: commands.Context, *, query: str):
            if not self.memory_store:
                return await ctx.reply("Память отключена или недоступна.")
            results = await self.memory_store.search(query, limit=5)
            if not results:
                return await ctx.reply("Нет записей по этому запросу.")
            lines = []
            for entry in results:
                when = datetime.fromtimestamp(entry.timestamp).strftime("%H:%M:%S")
                lines.append(f"[{when}] {entry.speaker}: {entry.text}")
            await ctx.reply("\n".join(lines))

        @ark_group.command(name="clear-log")
        async def ark_clear_log(ctx: commands.Context):
            if not self.memory_store:
                return await ctx.reply("Память уже отключена.")
            await self.memory_store.clear()
            await ctx.reply("Журнал памяти очищен.")

        # Load commands extension (prefix + slash)
        try:
            await self.load_extension("src.commands_cog")
        except Exception as e:
            logging.getLogger(__name__).error("Failed to load commands extension: %s", e)
        self._guild_id_env = os.environ.get("GUILD_ID")

    async def on_ready(self) -> None:
        logger = logging.getLogger(__name__)
        gid = getattr(self, "_guild_id_env", None) or os.environ.get("GUILD_ID")
        try:
            if gid:
                await self.sync_commands(guild_ids=[int(gid)], force=True)
            else:
                await self.sync_commands(force=True)
            pcmds = [c.name for c in self.commands]
            scmds = [c.name for c in self.application_commands]

            # Fallback: if no slash commands were registered through the Cog, add minimal ones here
            if not scmds:
                async def _join(ctx: discord.ApplicationContext):
                    if not ctx.author.voice or not ctx.author.voice.channel:
                        return await ctx.respond("Сначала подключись к голосовому каналу.", ephemeral=True)
                    channel = ctx.author.voice.channel
                    voice = discord.utils.get(self.voice_clients, guild=ctx.guild)
                    if voice and voice.is_connected():
                        await voice.move_to(channel)
                    else:
                        voice = await channel.connect()
                    await ctx.respond(f"Подключился к {channel.name}. Начинаю приём аудио.")
                    rt_model = os.environ.get("OPENAI_REALTIME_MODEL", "gpt-realtime")
                    fb_model = os.environ.get("OPENAI_WHISPER_MODEL") or self.campaign.get("asr", {}).get("model", "whisper-1")
                    self.voice_pipeline = VoicePipeline(
                        voice=voice,
                        router=self.router,
                        ark=self.ark_agent,
                        tts=self.tts,
                        openai_key=os.environ.get("OPENAI_API_KEY", ""),
                        rt_model=rt_model,
                        whisper_model=fb_model,
                        memory=self.memory_store,
                    )
                    await self.voice_pipeline.start()

                async def _leave(ctx: discord.ApplicationContext):
                    voice = discord.utils.get(self.voice_clients, guild=ctx.guild)
                    if voice and voice.is_connected():
                        if self.voice_pipeline:
                            self.voice_pipeline.stop()
                            self.voice_pipeline = None
                        await voice.disconnect()
                        await ctx.respond("Отключился от голосового канала.")
                    else:
                        await ctx.respond("Я не в голосовом канале.", ephemeral=True)

                self.add_application_command(discord.slash_command(name="join", description="Подключить бота к голосовому каналу")(_join))
                self.add_application_command(discord.slash_command(name="leave", description="Отключить бота от голосового канала")(_leave))
                # Sync once more after registering fallbacks
                if gid:
                    await self.sync_commands(guild_ids=[int(gid)], force=True)
                else:
                    await self.sync_commands(force=True)
                scmds = [c.name for c in self.application_commands]
            logger.info("Ready as %s. Prefix: %s | Slash: %s", self.user, pcmds, scmds)
        except Exception as e:
            logger.warning("on_ready sync failed: %s", e)

    async def close(self) -> None:
        if self.voice_pipeline:
            self.voice_pipeline.stop()
            self.voice_pipeline = None
        if self.memory_store:
            try:
                await self.memory_store.close()
            except Exception:
                logging.getLogger(__name__).debug("Failed to close memory store")
        await super().close()


def main() -> None:
    load_dotenv()
    setup_logging()
    ensure_opus_loaded()
    token = os.environ.get("DISCORD_TOKEN")
    if not token:
        raise RuntimeError("Set DISCORD_TOKEN in .env")
    bot = ARKBot()
    bot.run(token)


if __name__ == "__main__":
    main()
