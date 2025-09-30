"""Headless browser integration test for the Voice Changer frontend."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

from playwright.async_api import BrowserContext, Page, async_playwright

LOGGER = logging.getLogger(__name__)
DEFAULT_APP_URL = "http://localhost:5173/"
BROWSER_TEST_INPUT = Path(__file__).parent / "data" / "test_input.webm"


@dataclass(slots=True)
class BrowserTestConfig:
    """Runtime configuration values for the browser integration harness."""

    page_url: str = DEFAULT_APP_URL
    hold_seconds: float = 3.0
    connect_timeout: float = 15.0
    playback_timeout: float = 20.0
    record_output: Path | None = None
    record_seconds: float = 5.0

    @property
    def microphone_fixture(self) -> Path:
        """Return the audio fixture fed into Chromium's fake microphone.

        Returns:
            Filesystem location of the shared integration audio sample.
        """

        return BROWSER_TEST_INPUT


def _ensure_fixture(path: Path) -> None:
    """Validate that the audio fixture exists before launching Chromium.

    Args:
        path: Location of the audio fixture used for fake microphone capture.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Audio fixture missing at {path}. Ensure data/test_input.webm is available."
        )


async def _log_console(page: Page) -> None:
    """Attach console logging hooks so browser messages surface in stdout.

    Args:
        page: Page whose console events should be mirrored to the logger.
    """

    page.on(
        "console",
        lambda message: LOGGER.info("[browser] %s: %s", message.type, message.text),
    )
    page.on("pageerror", lambda error: LOGGER.error("[browser] page error: %s", error))


async def _await_selector(page: Page, selector: str, timeout: float) -> None:
    """Wait for the provided selector to appear on the page within the timeout.

    Args:
        page: Active Playwright page to query.
        selector: CSS selector that must become visible.
        timeout: Maximum seconds to wait for the selector to resolve.
    """

    await page.wait_for_selector(selector, timeout=timeout * 1000)


async def _await_button_text(page: Page, text: str, timeout: float) -> None:
    """Wait for the push-to-talk button to display the specified label.

    Args:
        page: Active Playwright page to query.
        text: Expected button label content.
        timeout: Maximum seconds to wait for the label to appear.
    """

    locator = page.get_by_role("button")
    await locator.filter(has_text=text).first.wait_for(timeout=timeout * 1000)


async def _trigger_press_and_hold(page: Page, hold_seconds: float) -> None:
    """Simulate pressing and holding the push-to-talk button.

    Args:
        page: Active Playwright page providing the button.
        hold_seconds: Seconds to keep the mouse button depressed.
    """

    button = page.get_by_role("button")
    await button.wait_for()
    box = await button.bounding_box()
    if not box:
        raise RuntimeError("Push-to-talk button does not have a bounding box")

    x = box["x"] + box["width"] / 2
    y = box["y"] + box["height"] / 2
    await page.mouse.move(x, y)
    await page.mouse.down()
    await asyncio.sleep(hold_seconds)


async def _release_button(page: Page) -> None:
    """Release the mouse button to stop streaming.

    Args:
        page: Active Playwright page whose mouse should be released.
    """

    await page.mouse.up()


async def _wait_for_remote_audio(page: Page, timeout: float) -> None:
    """Wait for a remote audio stream to attach to the tracked Audio element.

    Args:
        page: Active Playwright page to monitor for remote media.
        timeout: Maximum seconds to wait for the remote audio to appear.
    """

    await page.wait_for_function(
        """
        () => {
          const elements = window.__voiceChangerAudioElements || [];
          return elements.some((element) => element.srcObject && element.srcObject.getAudioTracks().length);
        }
        """,
        timeout=timeout * 1000,
    )


async def _record_remote_audio(page: Page, duration: float) -> bytes:
    """Capture remote audio playback into a WebM byte buffer.

    Args:
        page: Active Playwright page hosting the remote audio element.
        duration: Seconds of remote playback to capture.

    Returns:
        Raw audio/webm content produced by the MediaRecorder API.
    """

    return await page.evaluate(
        """
        async (durationSeconds) => {
          const elements = window.__voiceChangerAudioElements || [];
          const target = elements.find((element) => element.srcObject);
          if (!target) {
            throw new Error('Remote audio element not initialised');
          }

          const stream = target.captureStream();
          const recorder = new MediaRecorder(stream, { mimeType: 'audio/webm;codecs=opus' });
          const chunks = [];

          recorder.ondataavailable = (event) => {
            if (event.data && event.data.size) {
              chunks.push(event.data);
            }
          };

          const finished = new Promise((resolve, reject) => {
            recorder.onstop = () => resolve();
            recorder.onerror = (event) => reject(event.error || new Error('MediaRecorder error'));
          });

          recorder.start();
          await new Promise((resolve) => setTimeout(resolve, durationSeconds * 1000));
          recorder.stop();
          await finished;

          const blob = new Blob(chunks, { type: recorder.mimeType });
          const arrayBuffer = await blob.arrayBuffer();
          return Array.from(new Uint8Array(arrayBuffer));
        }
        """,
        duration,
    )


async def _run_test(config: BrowserTestConfig) -> None:
    """Execute the browser automation scenario and assert expected state transitions.

    Args:
        config: User-provided settings that drive the automation scenario.
    """

    _ensure_fixture(config.microphone_fixture)

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(
            headless=True,
            args=[
                "--use-fake-ui-for-media-stream",
                "--use-fake-device-for-media-stream",
                f"--use-file-for-fake-audio-capture={config.microphone_fixture}",
            ],
        )

        context: BrowserContext | None = None
        try:
            context = await browser.new_context()
            await context.grant_permissions(["microphone"], origin=config.page_url)
            await context.add_init_script(
                """
                (() => {
                  const created = [];
                  const OriginalAudio = window.Audio;
                  function TrackingAudio(...args) {
                    const element = new OriginalAudio(...args);
                    created.push(element);
                    window.__voiceChangerAudioElements = created;
                    return element;
                  }
                  TrackingAudio.prototype = OriginalAudio.prototype;
                  Object.setPrototypeOf(TrackingAudio, OriginalAudio);
                  window.Audio = TrackingAudio;
                })();
                """
            )

            page = await context.new_page()
            await _log_console(page)
            await page.goto(config.page_url, wait_until="networkidle")

            await _await_selector(page, ".connection-status.disconnected", timeout=10)
            await _trigger_press_and_hold(page, config.hold_seconds)
            await _await_button_text(page, "Streaming voice… release to stop", timeout=5)
            await _await_selector(page, ".connection-status.connecting", timeout=config.connect_timeout)
            await _await_selector(page, ".connection-status.connected", timeout=config.connect_timeout)
            await _wait_for_remote_audio(page, config.playback_timeout)

            if config.record_output:
                raw_bytes = await _record_remote_audio(page, config.record_seconds)
                config.record_output.write_bytes(bytes(raw_bytes))
                LOGGER.info("Saved remote playback to %s", config.record_output)

            await _release_button(page)
            await _await_selector(page, ".connection-status.disconnected", timeout=10)

            error_locator = page.locator(".error-message")
            if await error_locator.is_visible():
                raise RuntimeError(await error_locator.inner_text())
        finally:
            if context:
                await context.close()
            await browser.close()


def _build_parser() -> argparse.ArgumentParser:
    """Create the command-line interface for the integration test runner.

    Returns:
        Configured parser for CLI arguments.
    """

    parser = argparse.ArgumentParser(description="Run the headless browser integration test")
    parser.add_argument(
        "--url",
        default=DEFAULT_APP_URL,
        help="Base URL where the Vite dev server is accessible (default: %(default)s)",
    )
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=3.0,
        help="How long to hold the push-to-talk button before releasing",
    )
    parser.add_argument(
        "--connect-timeout",
        type=float,
        default=15.0,
        help="Maximum seconds to wait for the WebRTC connection to report 'connected'",
    )
    parser.add_argument(
        "--playback-timeout",
        type=float,
        default=20.0,
        help="Maximum seconds to wait for remote audio playback to start",
    )
    parser.add_argument(
        "--record-output",
        type=Path,
        default=None,
        help="Optional path to write remote playback as audio/webm",
    )
    parser.add_argument(
        "--record-seconds",
        type=float,
        default=5.0,
        help="Duration of remote playback capture when --record-output is set",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the browser integration test CLI.

    Args:
        argv: Optional argument vector overriding ``sys.argv``.

    Returns:
        Process exit status code, ``0`` on success.
    """

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = _build_parser()
    args = parser.parse_args(argv)

    config = BrowserTestConfig(
        page_url=args.url,
        hold_seconds=args.hold_seconds,
        connect_timeout=args.connect_timeout,
        playback_timeout=args.playback_timeout,
        record_output=args.record_output,
        record_seconds=args.record_seconds,
    )

    try:
        asyncio.run(_run_test(config))
    except Exception as exc:  # noqa: BLE001
        LOGGER.error("Browser integration test failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
