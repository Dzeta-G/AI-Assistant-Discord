from __future__ import annotations

import os
from datetime import datetime
from typing import Optional

import discord
from discord.ext import commands

from .discord_voice import VoicePipeline
from .ark_agent import ARKAgent
from .router import Router
from .tts_realtime import TTSClient


GID = os.getenv("GUILD_ID")
GUILD_IDS = [int(GID)] if GID and GID.isdigit() else None


def slash(name: str, description: str):
    def decorator(func):
        base = (
            discord.slash_command(name=name, description=description, guild_ids=GUILD_IDS)
            if GUILD_IDS
            else discord.slash_command(name=name, description=description)
        )
        return base(func)
    return decorator


class ARKCommands(commands.Cog):
    def __init__(self, bot: commands.Bot):
        self.bot = bot

    # ---------- Helpers ----------
    async def _send(self, ctx, message: str, ephemeral: bool = False):
        # Prefix context: reply; Slash context: respond
        if hasattr(ctx, "respond"):
            # Slash (ApplicationContext)
            await ctx.respond(message, ephemeral=ephemeral)
        else:
            await ctx.reply(message)

    async def _ensure_pipeline(self, guild: discord.Guild, voice: discord.VoiceClient) -> VoicePipeline:
        bot = self.bot
        assert isinstance(bot, commands.Bot)
        router: Router = getattr(bot, "router")
        ark: ARKAgent = getattr(bot, "ark_agent")
        tts: Optional[TTSClient] = getattr(bot, "tts")
        vp = VoicePipeline(
            voice=voice,
            router=router,
            ark=ark,
            tts=tts,
            openai_key=os.environ.get("OPENAI_API_KEY", ""),
            rt_model=os.environ.get("OPENAI_REALTIME_MODEL", "gpt-realtime"),
            whisper_model=os.environ.get("OPENAI_WHISPER_MODEL") or getattr(getattr(bot, "campaign", {}), "get", lambda *_: None)("asr", {}).get("model", "whisper-1"),
            memory=getattr(bot, "memory_store", None),
        )
        return vp

    # ---------- Prefix commands ----------
    @commands.command(name="join")
    async def join_cmd(self, ctx: commands.Context):
        if not ctx.author.voice or not ctx.author.voice.channel:
            return await self._send(ctx, "Сначала подключись к голосовому каналу.")
        channel = ctx.author.voice.channel
        voice = discord.utils.get(self.bot.voice_clients, guild=ctx.guild)
        if voice and voice.is_connected():
            await voice.move_to(channel)
        else:
            voice = await channel.connect()
        await self._send(ctx, f"Подключился к {channel.name}. Начинаю приём аудио.")
        # Start pipeline
        vp = await self._ensure_pipeline(ctx.guild, voice)
        self.bot.voice_pipeline = vp
        await vp.start()

    @commands.command(name="leave")
    async def leave_cmd(self, ctx: commands.Context):
        voice = discord.utils.get(self.bot.voice_clients, guild=ctx.guild)
        if voice and voice.is_connected():
            if getattr(self.bot, "voice_pipeline", None):
                self.bot.voice_pipeline.stop()
                self.bot.voice_pipeline = None
            await voice.disconnect()
            await self._send(ctx, "Отключился от голосового канала.")
        else:
            await self._send(ctx, "Я не в голосовом канале.")

    @commands.command(name="set-character")
    async def set_character_cmd(self, ctx: commands.Context, member: discord.Member, *, character: str):
        self.bot.router.set_character(member.id, character)
        await self._send(ctx, f"Назначен персонаж для {member.mention}: {character}")

    @commands.group(name="ark", invoke_without_command=True)
    async def ark_group_cmd(self, ctx: commands.Context):
        await self._send(ctx, "Используй: !ark on | !ark off | !ark wakeword <слово> | !ark test | !ark recall <запрос> | !ark clear-log")

    @ark_group_cmd.command(name="on")
    async def ark_on_cmd(self, ctx: commands.Context):
        self.bot.router.set_ark_enabled(True)
        await self._send(ctx, "ARK активирован.")

    @ark_group_cmd.command(name="off")
    async def ark_off_cmd(self, ctx: commands.Context):
        self.bot.router.set_ark_enabled(False)
        await self._send(ctx, "ARK отключен.")

    @ark_group_cmd.command(name="wakeword")
    async def ark_wakeword_cmd(self, ctx: commands.Context, *, word: str):
        self.bot.router.set_wake_word(word)
        await self._send(ctx, f"Wake-word установлен: {word}")

    @ark_group_cmd.command(name="test")
    async def ark_test_cmd(self, ctx: commands.Context, *, text: str = "Проверка связи."):
        reply = await self.bot.ark_agent.reply(self.bot.router.context(), text, None)
        if getattr(self.bot, "tts", None):
            try:
                path = await self.bot.tts.speak_to_tempfile(reply)
                voice = discord.utils.get(self.bot.voice_clients, guild=ctx.guild)
                if not voice or not voice.is_connected():
                    await self._send(ctx, reply)
                else:
                    from .discord_voice import VoicePipeline as _VP

                    vp = await self._ensure_pipeline(ctx.guild, voice)
                    await vp._play_file(path)
            except Exception:
                await self._send(ctx, reply)
        else:
            await self._send(ctx, reply)

    @ark_group_cmd.command(name="recall")
    async def ark_recall_cmd(self, ctx: commands.Context, *, query: str):
        store = getattr(self.bot, "memory_store", None)
        if not store:
            return await self._send(ctx, "Память отключена или недоступна.")
        results = await store.search(query, limit=5)
        if not results:
            return await self._send(ctx, "Нет записей по этому запросу.")
        lines = []
        for entry in results:
            when = datetime.fromtimestamp(entry.timestamp).strftime("%H:%M:%S")
            lines.append(f"[{when}] {entry.speaker}: {entry.text}")
        await self._send(ctx, "\n".join(lines))

    @ark_group_cmd.command(name="clear-log")
    async def ark_clear_log_cmd(self, ctx: commands.Context):
        store = getattr(self.bot, "memory_store", None)
        if not store:
            return await self._send(ctx, "Память уже отключена.")
        await store.clear()
        await self._send(ctx, "Журнал памяти очищен.")

    # ---------- Slash commands ----------
    @commands.command(name="ping")
    async def ping_cmd(self, ctx: commands.Context):
        await self._send(ctx, "pong")

    @slash("ping", "Проверка доступности бота")
    async def ping_slash(self, ctx: discord.ApplicationContext):
        await self._send(ctx, "pong", ephemeral=True)

    @slash("join", "Подключить бота к вашему голосовому каналу")
    async def join_slash(self, ctx: discord.ApplicationContext):
        await self.join_cmd(ctx)  # reuse implementation

    @slash("leave", "Отключить бота от голосового канала")
    async def leave_slash(self, ctx: discord.ApplicationContext):
        await self.leave_cmd(ctx)

    @slash("set_character", "Назначить персонажа пользователю")
    async def set_character_slash(self, ctx: discord.ApplicationContext, member: discord.Member, character: str):
        await self.set_character_cmd(ctx, member, character=character)

    @slash("ark_on", "Включить ARK")
    async def ark_on_slash(self, ctx: discord.ApplicationContext):
        await self.ark_on_cmd(ctx)

    @slash("ark_off", "Выключить ARK")
    async def ark_off_slash(self, ctx: discord.ApplicationContext):
        await self.ark_off_cmd(ctx)

    @slash("ark_wakeword", "Изменить wake-word")
    async def ark_wakeword_slash(self, ctx: discord.ApplicationContext, word: str):
        await self.ark_wakeword_cmd(ctx, word=word)

    @slash("ark_test", "Тестовый ответ ARK")
    async def ark_test_slash(self, ctx: discord.ApplicationContext, text: str = "Проверка связи."):
        await self.ark_test_cmd(ctx, text=text)

    @slash("ark_recall", "Поиск в памяти ARK")
    async def ark_recall_slash(self, ctx: discord.ApplicationContext, query: str):
        await self.ark_recall_cmd(ctx, query=query)

    @slash("ark_clear_log", "Очистить память ARK")
    async def ark_clear_log_slash(self, ctx: discord.ApplicationContext):
        await self.ark_clear_log_cmd(ctx)


async def setup(bot: commands.Bot):
    # Extension entrypoint for py-cord
    await bot.add_cog(ARKCommands(bot))
