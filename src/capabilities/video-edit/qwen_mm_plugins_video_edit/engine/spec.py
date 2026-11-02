"""Edit spec: the single contract between whoever decides the edit and the deterministic renderer."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

# xfade transition names accepted by FFmpeg's xfade filter (a stable subset).
XfadeName = Literal[
    "fade",
    "fadeblack",
    "fadewhite",
    "dissolve",
    "wipeleft",
    "wiperight",
    "wipeup",
    "wipedown",
    "slideleft",
    "slideright",
    "slideup",
    "slidedown",
    "circleopen",
    "circleclose",
    "radial",
    "smoothleft",
    "smoothright",
    "zoomin",
    "pixelize",
]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Transition(_Strict):
    type: Literal["cut"] | XfadeName = Field("cut", description="'cut' or an FFmpeg xfade transition name")
    duration: float = Field(0.5, gt=0, le=5, description="transition length in seconds (ignored for 'cut')")


class Clip(_Strict):
    src: str = Field(description="path to the source video or still image")
    start: float = Field(0.0, ge=0, description="in-point in the source, seconds")
    end: Optional[float] = Field(None, gt=0, description="out-point in the source, seconds; default = source end")
    speed: float = Field(1.0, ge=0.25, le=4.0, description="playback speed multiplier")
    volume: float = Field(1.0, ge=0, le=4.0, description="gain applied to this clip's own audio (0 mutes)")
    fade_in: float = Field(0.0, ge=0, description="fade from black at the clip head, seconds")
    fade_out: float = Field(0.0, ge=0, description="fade to black at the clip tail, seconds")
    transition: Transition = Field(default_factory=Transition, description="transition INTO the next clip")

    @model_validator(mode="after")
    def _ordered(self) -> Clip:
        if self.end is not None and self.end <= self.start:
            raise ValueError(f"clip end ({self.end}) must be greater than start ({self.start})")
        return self


class Caption(_Strict):
    text: str = Field(min_length=1)
    start: float = Field(ge=0, description="seconds on the OUTPUT timeline")
    end: float = Field(gt=0, description="seconds on the OUTPUT timeline")

    @model_validator(mode="after")
    def _ordered(self) -> Caption:
        if self.end <= self.start:
            raise ValueError("caption end must be greater than start")
        return self


class CaptionStyle(_Strict):
    font: str = "Arial"
    size: int = Field(14, ge=6, le=96, description="libass font size (relative to a 288-line canvas)")
    position: Literal["bottom", "top", "center"] = "bottom"
    outline: float = Field(1.5, ge=0, le=6)


class Music(_Strict):
    src: str
    start: float = Field(0.0, ge=0, description="offset into the music file, seconds")
    volume: float = Field(0.3, ge=0, le=4.0)
    fade_in: float = Field(0.0, ge=0)
    fade_out: float = Field(2.0, ge=0)
    duck: bool = Field(True, description="sidechain-duck the music under the clips' own audio")


class Output(_Strict):
    path: str
    width: int = Field(1920, ge=16, le=7680)
    height: int = Field(1080, ge=16, le=4320)
    fps: float = Field(30.0, gt=0, le=240)
    encoder: Literal["auto", "libx264", "h264_nvenc", "hevc_nvenc", "libx265"] = Field(
        "auto", description="'auto' uses NVENC when a working NVIDIA GPU is present, else libx264"
    )
    crf: int = Field(20, ge=0, le=51, description="quality (CRF for x264/x265, CQ for NVENC)")
    preset: Optional[str] = Field(None, description="encoder preset; default picks a sensible one per encoder")
    audio_bitrate: str = "192k"
    loudness: Optional[float] = Field(-14.0, ge=-40, le=-5, description="integrated LUFS target; null skips")

    @model_validator(mode="after")
    def _even(self) -> Output:
        if self.width % 2 or self.height % 2:
            raise ValueError("output width and height must be even for yuv420p")
        return self


class EditSpec(_Strict):
    """A complete, declarative edit. Rendering it is deterministic."""

    output: Output
    clips: list[Clip] = Field(min_length=1)
    captions: list[Caption] = Field(default_factory=list)
    caption_style: CaptionStyle = Field(default_factory=CaptionStyle)
    subtitles_file: Optional[str] = Field(None, description="existing .srt/.ass to burn in (instead of captions)")
    music: Optional[Music] = None

    @model_validator(mode="after")
    def _one_subtitle_source(self) -> EditSpec:
        if self.captions and self.subtitles_file:
            raise ValueError("use either captions or subtitles_file, not both")
        return self
