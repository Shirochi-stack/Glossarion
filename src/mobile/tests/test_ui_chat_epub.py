"""Device UI test: a chat translates the self-test EPUB end to end (``flet test android``; U9).

Settings › Import from desktop (config.json → the test's fake model server through adb reverse),
a new chat, ＋ › Files (the system picker) with the 12-chapter Korean EPUB, Send, the job card
reaches Done, the book moves into the Library by itself (auto-migrate, "Added to the Library"), then
Library › the book › Chapters shows every chapter completed. The same
flow runs on the host in tests_host/test_ui_flows.py.
"""

import flows


async def test_chat_epub_to_library(ui, device_files, fake_server):
    # every device test is a fresh install: wait for the Welcome (pushed after the home) and skip it
    await flows.dismiss_welcome(ui, timeout=180, first_run=True)
    await flows.wait_home(ui)
    await flows.import_desktop_config(ui)
    await flows.chat_translate_and_migrate(ui)
    assert fake_server.requests, "the app never reached the fake model server"
    await flows.library_book_chapters(ui)
