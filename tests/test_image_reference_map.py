import io
import json
import zipfile
from pathlib import Path

import pytest
from bs4 import BeautifulSoup

import image_reference_map as tracking


def inventory(folder, images, rename_map=None, pages=None):
    """Simulate extraction's source bytes and successfully renamed resources."""
    image_dir = folder / 'images'
    image_dir.mkdir(exist_ok=True)
    for path in image_dir.iterdir():
        if path.is_file():
            path.unlink()
    rename_map = rename_map or {}
    (folder / 'image_rename_map.json').write_text(json.dumps(rename_map), encoding='utf-8')
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as archive:
        for source, contents in images.items():
            archive.writestr(source, contents)
            filename = rename_map.get(Path(source).name, Path(source).name)
            (image_dir / filename).write_bytes(contents)
        for source, markup in (pages or {}).items():
            archive.writestr(source, markup)
    buffer.seek(0)
    with zipfile.ZipFile(buffer) as archive:
        tracking.publish_extracted_images(
            folder, archive, lambda member, name: ('images', 'images', name)
            if Path(member).suffix.lower() in tracking.IMAGE_EXTENSIONS else None)


def page(folder, name, refs):
    path = folder / name
    path.write_text('<html><body>' + ''.join(f'<img src="images/{ref}">' for ref in refs) + '</body></html>', encoding='utf-8')
    return path


def sources(path):
    return [tag['src'] for tag in BeautifulSoup(path.read_text(encoding='utf-8'), 'html.parser').find_all('img')]


sources_of = sources


def state(folder):
    return json.loads((folder / tracking.SIDECAR).read_text(encoding='utf-8'))


def test_shared_gallery_and_chapters_keep_order_and_one_parent(tmp_path):
    inventory(tmp_path, {'OEBPS/Images/a.jpg': b'A', 'OEBPS/Images/b.jpg': b'B'},
              {'a.jpg': 'Gallery_img_1.jpg', 'b.jpg': 'Gallery_img_2.jpg'},
              {'OEBPS/Text/Gallery.xhtml': '<img src="../Images/a.jpg"><img src="../Images/b.jpg">',
               'OEBPS/Text/Section0005.xhtml': '<img src="../Images/a.jpg">'})
    gallery = page(tmp_path, 'response_Gallery.xhtml', ['Gallery_img_1.jpg', 'Gallery_img_2.jpg'])
    chapter = page(tmp_path, 'response_Section0005.xhtml', ['Gallery_img_2.jpg', 'Gallery_img_1.jpg', 'Gallery_img_1.jpg'])
    other = page(tmp_path, 'response_Section0009.xhtml', ['Gallery_img_1.jpg'])
    original = {p.name: p.read_bytes() for p in (gallery, chapter, other)}
    assert tracking.repair_image_references(tmp_path) == 0
    assert {p.name: p.read_bytes() for p in (gallery, chapter, other)} == original
    parent = state(tmp_path)['parents'][state(tmp_path)['current']['Gallery_img_1.jpg']]
    assert parent['referencing_pages'] == sorted(original)
    assert parent['source_pages'] == ['OEBPS/Text/Gallery.xhtml', 'OEBPS/Text/Section0005.xhtml']
    assert sorted(p.name for p in (tmp_path / 'images').iterdir()) == ['Gallery_img_1.jpg', 'Gallery_img_2.jpg']


def test_inserted_chapter_and_reused_filename_repair_by_bytes(tmp_path):
    inventory(tmp_path, {'OEBPS/Images/ill5.jpg': b'old-illustration'}, {'ill5.jpg': 'Section0005_img_1.jpg'})
    pages = [page(tmp_path, name, ['Section0005_img_1.jpg']) for name in
             ('response_Gallery.xhtml', 'response_Section0005.xhtml', 'response_Section0008.xhtml')]
    tracking.snapshot_before_refresh(tmp_path)
    inventory(tmp_path, {'OEBPS/Images/ill6.jpg': b'old-illustration', 'OEBPS/Images/ill5.jpg': b'new-illustration'},
              {'ill6.jpg': 'Section0006_img_1.jpg', 'ill5.jpg': 'Section0005_img_1.jpg'})
    assert tracking.repair_image_references(tmp_path) == 3
    assert all(sources(p) == ['images/Section0006_img_1.jpg'] for p in pages)
    before = {p: p.read_bytes() for p in [*pages, tmp_path / tracking.SIDECAR, tmp_path / 'image_rename_map.json']}
    assert tracking.repair_image_references(tmp_path) == 0
    assert all(p.read_bytes() == contents for p, contents in before.items())


@pytest.mark.parametrize('new_images', [{}, {'OEBPS/Images/a.jpg': b'changed'}])
def test_deleted_or_changed_image_is_unresolved_across_retries(tmp_path, new_images):
    inventory(tmp_path, {'OEBPS/Images/a.jpg': b'original'})
    output = page(tmp_path, 'response_Chapter.xhtml', ['a.jpg'])
    tracking.snapshot_before_refresh(tmp_path)
    inventory(tmp_path, new_images)
    log = []
    original = output.read_bytes()
    assert tracking.repair_image_references(tmp_path, log=log.append) == 0
    assert any('Unresolved' in line for line in log)
    old_identity = state(tmp_path)['pages'][output.name]['a.jpg']
    tracking.snapshot_before_refresh(tmp_path)
    inventory(tmp_path, {'OEBPS/Images/restored.jpg': b'original'}, {'restored.jpg': 'Chapter0002_img_1.jpg'})
    assert old_identity in state(tmp_path)['parents']
    assert tracking.repair_image_references(tmp_path) == 1
    assert sources(output) == ['images/Chapter0002_img_1.jpg']
    assert original != output.read_bytes()


def test_reencoded_image_is_repaired_to_its_new_extension(tmp_path):
    inventory(tmp_path, {'OEBPS/Images/ch5-0c84.jpg': b'jpeg-bytes', 'OEBPS/Images/ch6-44d3.jpg': b'jpeg-six'},
              {'ch5-0c84.jpg': 'chapter0005_img_1.jpg', 'ch6-44d3.jpg': 'chapter0006_img_1.jpg'})
    output = tmp_path / 'response_chapter0005.html'
    output.write_text('<html><body><img alt="ch5-0c84.jpg" src="../Images/chapter0005_img_1.jpg"/>'
                      '<img src="../Images/chapter0006_img_1.jpg"/></body></html>', encoding='utf-8')
    tracking.snapshot_before_refresh(tmp_path)
    inventory(tmp_path, {'OEBPS/Images/ch5-0c84.webp': b'webp-bytes', 'OEBPS/Images/ch6-44d3.webp': b'webp-six'},
              {'ch5-0c84.webp': 'chapter0005_img_1.webp', 'ch6-44d3.webp': 'chapter0006_img_1.webp'})
    assert tracking.repair_image_references(tmp_path) == 1
    # ``../Images/`` is the source EPUB's folder; the workspace keeps images in ``images/``.
    assert sources(output) == ['images/chapter0005_img_1.webp', 'images/chapter0006_img_1.webp']
    assert all((tmp_path / source).is_file() for source in sources(output))
    assert 'alt="ch5-0c84.jpg"' in output.read_text(encoding='utf-8')
    before = output.read_bytes()
    assert tracking.repair_image_references(tmp_path) == 0
    assert output.read_bytes() == before


def test_right_name_in_a_folder_that_does_not_exist_moves_to_images(tmp_path):
    inventory(tmp_path, {'OEBPS/Images/a.webp': b'A'}, {'a.webp': 'chapter0001_img_1.webp'})
    output = tmp_path / 'response_chapter0001.html'
    output.write_text('<html><body><img src="../Images/chapter0001_img_1.webp?v=1#top"/>'
                      '<img src="images/chapter0001_img_1.webp"/>'
                      '<img src="https://example.com/Images/chapter0001_img_1.webp"/></body></html>', encoding='utf-8')
    log = []
    assert tracking.repair_image_references(tmp_path, log=log.append) == 1
    assert sources(output) == ['images/chapter0001_img_1.webp?v=1#top', 'images/chapter0001_img_1.webp',
                               'https://example.com/Images/chapter0001_img_1.webp']
    assert sum('Repaired image folder' in line for line in log) == 1
    assert tracking.repair_image_references(tmp_path) == 0


def test_a_written_page_names_images_and_the_next_repair_leaves_it(tmp_path):
    from html_output_utils import write_utf8_html_file

    inventory(tmp_path, {'OEBPS/Images/a.webp': b'A'}, {'a.webp': 'chapter0001_img_1.webp'})
    output = tmp_path / 'response_chapter0001.html'
    write_utf8_html_file(str(output), '<html><body><img src="../Images/chapter0001_img_1.webp?v=1#top"/>'
                                      '<img src="images/chapter0001_img_1.webp"/>'
                                      '<img src="../Images/missing.webp"/>'
                                      '<img src="https://example.com/Images/chapter0001_img_1.webp"/></body></html>')
    # the link is right from the write; only a file that images/ holds is moved there
    assert sources(output) == ['images/chapter0001_img_1.webp?v=1#top', 'images/chapter0001_img_1.webp',
                               '../Images/missing.webp', 'https://example.com/Images/chapter0001_img_1.webp']
    before = output.read_bytes()
    log = []
    assert tracking.repair_image_references(tmp_path, log=log.append) == 0
    assert output.read_bytes() == before and not any('Repaired' in line for line in log)


def test_a_written_page_without_a_folder_to_fix_keeps_its_text(tmp_path):
    from html_output_utils import ensure_utf8_html_document, write_utf8_html_file

    inventory(tmp_path, {'OEBPS/Images/a.webp': b'A'})
    for name, markup in (('plain.html', '<html><body><p>No picture &amp; no link<br></p></body></html>'),
                         ('right.html', '<html><body><img src="images/a.webp"><br></body></html>'),
                         ('elsewhere.html', '<html><body><img src="../Images/b.webp"><br></body></html>')):
        write_utf8_html_file(str(tmp_path / name), markup)
        assert (tmp_path / name).read_text(encoding='utf-8') == ensure_utf8_html_document(markup)


def test_reencoded_repair_does_not_guess_between_two_formats(tmp_path):
    inventory(tmp_path, {'OEBPS/Images/a.jpg': b'original'})
    output = page(tmp_path, 'response_Chapter.xhtml', ['a.jpg'])
    tracking.snapshot_before_refresh(tmp_path)
    inventory(tmp_path, {'OEBPS/Images/a.webp': b'one', 'OEBPS/Images/a.png': b'two'})
    log = []
    assert tracking.repair_image_references(tmp_path, log=log.append) == 0
    assert sources(output) == ['images/a.jpg']
    assert any('Unresolved' in line for line in log)


def test_resolve_qa_repairs_links_first_and_inserts_only_missing_images(tmp_path):
    import progress_actions

    folder = tmp_path / 'book'
    folder.mkdir()
    inventory(folder, {'OEBPS/Images/a-1.jpg': b'jpeg-one', 'OEBPS/Images/a-2.jpg': b'jpeg-two'},
              {'a-1.jpg': 'chapter0001_img_1.jpg', 'a-2.jpg': 'chapter0001_img_2.jpg'})
    both = page(folder, 'response_chapter0001.html', ['chapter0001_img_1.jpg', 'chapter0001_img_2.jpg'])
    one = page(folder, 'response_chapter0002.html', ['chapter0001_img_1.jpg'])
    untouched = page(folder, 'response_chapter0003.html', ['chapter0001_img_1.jpg'])
    tracking.snapshot_before_refresh(folder)
    inventory(folder, {'OEBPS/Images/a-1.webp': b'webp-one', 'OEBPS/Images/a-2.webp': b'webp-two'},
              {'a-1.webp': 'chapter0001_img_1.webp', 'a-2.webp': 'chapter0001_img_2.webp'})
    source = '<html><body><img src="../Images/a-1.webp"/><img src="../Images/a-2.webp"/></body></html>'
    epub = tmp_path / 'source.epub'
    with zipfile.ZipFile(epub, 'w') as archive:
        for name in ('chapter0001', 'chapter0002', 'chapter0003'):
            archive.writestr(f'OEBPS/Text/{name}.xhtml', source)
    entries = {
        '1': {'output_file': both.name, 'status': 'qa_failed', 'qa_issues': True,
              'qa_issues_found': ['changed_image_references_2_found', 'Chinese_text_found_2_chars'],
              'qa_issue_previews': {'changed_image_references_2_found': 'chapter0001_img_1.jpg'}},
        '2': {'output_file': one.name, 'status': 'qa_failed', 'qa_issues': True,
              'qa_issues_found': ['missing_images_1_lost_(1/2)']},
        '3': {'output_file': untouched.name, 'status': 'qa_failed', 'qa_issues': True,
              'qa_issues_found': ['Chinese_text_found_2_chars']},
    }
    progress = folder / 'translation_progress.json'
    progress.write_text(json.dumps({'chapters': entries}), encoding='utf-8')
    rows = [{'progress_key': key, 'output_file': entry['output_file'],
             'original_filename': entry['output_file'].replace('response_', '').replace('.html', '.xhtml'),
             'info': entry} for key, entry in entries.items()]
    restored = []

    def restore(translated, _source, verbose=True, rename_map=None):
        restored.append(translated)
        return translated.replace('</body>', '<img src="images/chapter0001_img_2.webp"/></body>')

    kind, _title, message, refreshed = progress_actions.resolve_image_qa(
        {'output_dir': str(folder), 'file_path': str(epub), 'progress_file': str(progress)},
        rows, restore_fn=restore, log=lambda _line: None)
    assert (kind, refreshed) == ('info', True)
    assert sources(both) == ['images/chapter0001_img_1.webp', 'images/chapter0001_img_2.webp']
    # The existing link was repaired before the insertion saw the file.
    assert len(restored) == 1 and 'chapter0001_img_1.webp' in restored[0]
    assert sources(one) == ['images/chapter0001_img_1.webp', 'images/chapter0001_img_2.webp']
    assert sources(untouched) == ['images/chapter0001_img_1.jpg']
    saved = json.loads(progress.read_text(encoding='utf-8'))['chapters']
    assert saved['1']['qa_issues_found'] == ['Chinese_text_found_2_chars']
    assert 'qa_issue_previews' not in saved['1'] and saved['1']['status'] == 'qa_failed'
    assert 'qa_issues_found' not in saved['2'] and saved['2']['status'] == 'completed'
    assert saved['3'] == entries['3']
    assert 'cleared on 2 of 2' in message


def test_resolve_qa_puts_pictures_back_inside_their_source_links(tmp_path):
    import progress_actions

    folder = tmp_path / 'book'
    folder.mkdir()
    inventory(folder, {'OEBPS/Images/a.webp': b'one', 'OEBPS/Images/b.webp': b'two'},
              {'a.webp': 'chapter0001_img_1.webp', 'b.webp': 'chapter0001_img_2.webp'})
    source = ('<html><body><p><a href="https://host.example/i1/"><img src="../Images/a.webp"/></a></p>'
              '<p><a href="https://host.example/i2/"><img src="../Images/b.webp"/></a></p>'
              '<p><a href="#note">note</a></p></body></html>')
    epub = tmp_path / 'source.epub'
    with zipfile.ZipFile(epub, 'w') as archive:
        for name in ('chapter0001', 'chapter0002', 'chapter0003'):
            archive.writestr(f'OEBPS/Text/{name}.xhtml', source)
    garbled = "<a href='&lt;https: host.example=\"\" i1=\"\"&gt;&lt;/https:&gt;'></a>"
    pictures = ('<p><img src="images/chapter0001_img_1.webp"/>' + garbled + '</p>'
                '<p><img src="images/chapter0001_img_2.webp"/></p>')
    fixed = folder / 'response_chapter0001.html'
    fixed.write_text(f'<html><body>{pictures}<p><a href="#note">chú</a></p></body></html>', encoding='utf-8')
    partly = folder / 'response_chapter0002.html'  # the note link is lost as well
    partly.write_text(f'<html><body>{pictures}<p>chú</p></body></html>', encoding='utf-8')
    chunk = folder / 'response_chapter0003.html'
    chunk.write_text(f'<html><body>{pictures}</body></html>', encoding='utf-8')
    chunk_before = chunk.read_bytes()
    entries = {
        '1': {'output_file': fixed.name, 'status': 'qa_failed', 'qa_issues': True,
              'qa_issues_found': ['changed_link_targets_2_found'],
              'qa_issue_previews': {'changed_link_targets_2_found': 'missing: https://host.example/i1/'}},
        '2': {'output_file': partly.name, 'status': 'qa_failed', 'qa_issues': True,
              'qa_issues_found': ['missing_link_targets_3_lost_(0/3)', 'Chinese_text_found_2_chars']},
        '3': {'output_file': chunk.name, 'status': 'qa_failed', 'qa_issues': True,
              'qa_issues_found': ['missing_link_targets_3_lost_(0/3)']},
    }
    progress = folder / 'translation_progress.json'
    progress.write_text(json.dumps({'chapters': entries}), encoding='utf-8')
    rows = [{'progress_key': key, 'output_file': entry['output_file'], 'is_chunk_progress': key == '3',
             'original_filename': entry['output_file'].replace('response_', '').replace('.html', '.xhtml'),
             'info': entry} for key, entry in entries.items()]
    data = {'output_dir': str(folder), 'file_path': str(epub), 'progress_file': str(progress)}

    kind, _title, message, refreshed = progress_actions.resolve_image_qa(data, rows, log=lambda _line: None)

    assert (kind, refreshed) == ('info', True)
    linked = ('<p><a href="https://host.example/i1/"><img src="images/chapter0001_img_1.webp"/></a></p>'
              '<p><a href="https://host.example/i2/"><img src="images/chapter0001_img_2.webp"/></a></p>')
    assert fixed.read_text(encoding='utf-8') == f'<html><body>{linked}<p><a href="#note">chú</a></p></body></html>'
    assert partly.read_text(encoding='utf-8') == f'<html><body>{linked}<p>chú</p></body></html>'
    assert chunk.read_bytes() == chunk_before  # a chunk row is left to its own marks
    saved = json.loads(progress.read_text(encoding='utf-8'))['chapters']
    assert 'qa_issues_found' not in saved['1'] and saved['1']['status'] == 'completed'
    assert saved['2'] == entries['2']  # one link is still lost: the mark stays
    assert saved['3'] == entries['3']
    assert 'Put 4 picture(s) back inside their links in 2 file(s).' in message
    assert 'Link QA issue cleared on 1 of 2' in message
    # nothing is left to do: the caller may go on to another resolution
    rows = [{**row, 'info': saved[row['progress_key']]} for row in rows]
    assert progress_actions.resolve_image_qa(data, rows, log=lambda _line: None)[3] is False


def test_resolve_qa_turns_a_link_the_source_does_not_have_back_into_text():
    import progress_actions

    source = ('<body><p>Notice</p><p>→https://host.example/n1/</p><p><a href="#note">note</a></p>'
              '<p>see https://host.example/shown/</p></body>')
    made_up = ('<body><p><a href="https://host.example/n1/">Notice</a></p><p><a href="#note">chú</a></p>'
               '<p>see <a href="https://host.example/shown/">https://host.example/shown/</a></p></body>')
    html, changed = progress_actions._unlink_added_links(made_up, source)
    assert changed == 2
    # the address the source shows stays readable; one already shown is not repeated
    assert html == ('<body><p>Notice https://host.example/n1/</p><p><a href="#note">chú</a></p>'
                    '<p>see https://host.example/shown/</p></body>')
    assert progress_actions._unlink_added_links(html, source) == (html, 0)
    # a source link is missing: the added link may be that link with a changed address
    changed_address = '<body><p>Notice</p><p><a href="#note-1">chú</a></p></body>'
    assert progress_actions._unlink_added_links(changed_address, source) == (changed_address, 0)


def test_run_repair_puts_pictures_back_inside_their_source_links(tmp_path):
    inventory(tmp_path, {'OEBPS/Images/a.webp': b'one', 'OEBPS/Images/b.webp': b'two'},
              {'a.webp': 'chapter0001_img_1.webp', 'b.webp': 'chapter0001_img_2.webp'})
    linked_source = ('<html><body><p><a href="https://host.example/i1/"><img src="../Images/a.webp"/></a></p>'
                     '<p><a class="x" href="other.xhtml#top"><span><img src="../Images/b.webp"/></span></a></p>'
                     '</body></html>')
    plain_source = '<html><body><p><img src="../Images/a.webp"/></p><p><a href="#n">n</a></p></body></html>'
    epub = tmp_path / 'source.epub'
    with zipfile.ZipFile(epub, 'w') as archive:
        archive.writestr('OEBPS/Text/chapter0001.xhtml', linked_source)
        archive.writestr('OEBPS/Text/chapter0002.xhtml', linked_source)
        archive.writestr('OEBPS/Text/chapter0003.xhtml', plain_source)
    (tmp_path / 'source_epub.txt').write_text(str(epub), encoding='utf-8')
    lost = tmp_path / 'response_chapter0001.html'
    lost.write_text('<html><body><p><img src="images/chapter0001_img_1.webp"/><a href="x"></a></p>'
                    '<p><img src="images/chapter0001_img_2.webp"/></p></body></html>', encoding='utf-8')
    uneven = tmp_path / 'response_chapter0002.html'  # one picture is gone: nothing can be paired
    uneven.write_text('<html><body><p><img src="images/chapter0001_img_1.webp"/></p></body></html>', encoding='utf-8')
    plain = tmp_path / 'response_chapter0003.html'
    plain.write_text('<html><body><p><img src="images/chapter0001_img_1.webp"/></p><p><a href="#n">n</a></p></body></html>',
                     encoding='utf-8')
    untouched = {path: path.read_bytes() for path in (uneven, plain)}
    log = []

    assert tracking.repair_image_references(tmp_path, log=log.append) == 1

    assert lost.read_text(encoding='utf-8') == (
        '<html><body><p><a href="https://host.example/i1/"><img src="images/chapter0001_img_1.webp"/></a></p>'
        '<p><a href="other.xhtml#top"><img src="images/chapter0001_img_2.webp"/></a></p></body></html>')
    assert all(path.read_bytes() == contents for path, contents in untouched.items())
    assert [line for line in log if 'picture' in line] == [
        'Put 2 picture(s) back inside their links in response_chapter0001.html']
    assert tracking.repair_image_references(tmp_path, log=log.append) == 0
    # a repair limited to named files (Resolve QA issue) leaves the pictures to its caller
    lost.write_text(lost.read_text(encoding='utf-8').replace('<a href="https://host.example/i1/">', '').replace(
        '.webp"/></a></p><p><a', '.webp"/></p><p><a'), encoding='utf-8')
    assert tracking.repair_image_references(tmp_path, only={lost.name}) == 0


def test_stale_extraction_is_refreshed_only_when_the_marker_names_another_epub(tmp_path, monkeypatch):
    import Chapter_Extractor as extractor

    epub = tmp_path / 'source.epub'
    with zipfile.ZipFile(epub, 'w') as archive:
        archive.writestr('OEBPS/Text/chapter0001.xhtml', '<html><body><p>one</p></body></html>')
    folder = tmp_path / 'book'
    folder.mkdir()
    calls = []
    monkeypatch.setattr(extractor, 'extract_chapters', lambda archive, output_dir, **_kwargs: calls.append(output_dir))
    verdict = {'valid': True}
    monkeypatch.setattr(extractor, '_validate_resource_extraction_marker',
                        lambda *_args: (verdict['valid'], '' if verdict['valid'] else 'source EPUB content changed'))
    log = []

    # no marker: nothing says which EPUB the images came from
    assert tracking.refresh_stale_extraction(str(folder), str(epub), log.append) is False
    (folder / '.resources_extracted').write_text('{}', encoding='utf-8')
    assert tracking.refresh_stale_extraction(str(folder), str(epub), log.append) is False
    assert tracking.refresh_stale_extraction(str(folder), str(tmp_path / 'missing.epub'), log.append) is False
    assert tracking.refresh_stale_extraction(str(folder), str(tmp_path / 'source.pdf'), log.append) is False
    assert calls == [] and log == []
    verdict['valid'] = False
    assert tracking.refresh_stale_extraction(str(folder), str(epub), log.append) is True
    assert calls == [str(folder)] and len(log) == 1


def test_qa_scan_auto_repair_touches_only_the_pages_it_would_flag(tmp_path):
    import scan_html_folder as scanner

    inventory(tmp_path, {'OEBPS/Images/a.jpg': b'jpeg-one'}, {'a.jpg': 'chapter0001_img_1.jpg'})
    renamed = page(tmp_path, 'response_chapter0001.html', ['chapter0001_img_1.jpg'])
    tracking.snapshot_before_refresh(tmp_path)
    inventory(tmp_path, {'OEBPS/Images/a.webp': b'webp-one'}, {'a.webp': 'chapter0001_img_1.webp'})
    sources = {
        'chapter0001.xhtml': '<html><body><p><img src="../Images/a.webp"/></p></body></html>',
        'chapter0002.xhtml': '<html><body><p><a href="https://host.example/i1/"><img src="../Images/a.webp"/></a></p></body></html>',
        'chapter0003.xhtml': '<html><body><p><img src="../Images/a.webp"/></p></body></html>',
    }
    epub = tmp_path / 'source.epub'
    with zipfile.ZipFile(epub, 'w') as archive:
        for name, markup in sources.items():
            archive.writestr(f'OEBPS/Text/{name}', markup)
    (tmp_path / 'source_epub.txt').write_text(str(epub), encoding='utf-8')
    unlinked = page(tmp_path, 'response_chapter0002.html', ['chapter0001_img_1.webp'])
    fine = page(tmp_path, 'response_chapter0003.html', ['chapter0001_img_1.webp'])
    fine_before = fine.read_bytes()
    rename_map = {'a.webp': 'chapter0001_img_1.webp'}
    image_info = {index: {'image_count': 1, 'image_srcs': ['../Images/a.webp'], 'filename': name}
                  for index, name in enumerate(sources)}
    preservation = {name: scanner._html_preservation_profile(markup, rename_map) for name, markup in sources.items()}
    names = [renamed.name, unlinked.name, fine.name]
    log = []

    # the link target check is off: only the page with the changed image reference is flagged
    assert scanner._auto_repair_flagged_pages(
        str(tmp_path), names, image_info, preservation, rename_map, False, log.append) == [renamed.name]
    assert sources_of(renamed) == ['images/chapter0001_img_1.webp']
    assert sources_of(unlinked) == ['images/chapter0001_img_1.webp'] and '<a ' not in unlinked.read_text(encoding='utf-8')
    assert fine.read_bytes() == fine_before

    # with it on, the page whose picture left its link is flagged and relinked
    assert scanner._auto_repair_flagged_pages(
        str(tmp_path), names, image_info, preservation, rename_map, True, log.append) == [unlinked.name]
    assert '<a href="https://host.example/i1/"><img src="images/chapter0001_img_1.webp"/></a>' in unlinked.read_text(
        encoding='utf-8')
    assert fine.read_bytes() == fine_before
    assert scanner._auto_repair_flagged_pages(
        str(tmp_path), names, image_info, preservation, rename_map, True, log.append) == []


def test_identical_bytes_use_source_path_and_ambiguous_matches_do_not_guess(tmp_path):
    inventory(tmp_path, {'OEBPS/Images/a.jpg': b'same', 'OEBPS/Images/b.jpg': b'same'})
    output = page(tmp_path, 'response_Chapter.xhtml', ['a.jpg', 'b.jpg'])
    tracking.snapshot_before_refresh(tmp_path)
    inventory(tmp_path, {'OEBPS/Images/a.jpg': b'same', 'OEBPS/Images/c.jpg': b'same'},
              {'a.jpg': 'Section0006_img_1.jpg', 'c.jpg': 'Section0007_img_1.jpg'})
    assert tracking.repair_image_references(tmp_path) == 1
    assert sources(output) == ['images/Section0006_img_1.jpg', 'images/b.jpg']
    assert state(tmp_path)['pages'][output.name]['b.jpg'] is not None


def test_snapshot_is_durable_and_not_rebuilt_from_partial_new_resources(tmp_path):
    inventory(tmp_path, {'OEBPS/Images/a.jpg': b'old'}, {'a.jpg': 'Chapter_img_1.jpg'})
    output = page(tmp_path, 'response_Chapter.xhtml', ['Chapter_img_1.jpg'])
    tracking.snapshot_before_refresh(tmp_path)
    before = (tmp_path / tracking.SIDECAR).read_bytes()
    (tmp_path / 'images' / 'Chapter_img_1.jpg').write_bytes(b'partial-new')
    tracking.snapshot_before_refresh(tmp_path)
    assert (tmp_path / tracking.SIDECAR).read_bytes() == before
    assert tracking.repair_image_references(tmp_path) == 0
    inventory(tmp_path, {'OEBPS/Images/b.jpg': b'old'}, {'b.jpg': 'Chapter2_img_1.jpg'})
    assert tracking.repair_image_references(tmp_path) == 1
    assert sources(output) == ['images/Chapter2_img_1.jpg']


def test_legacy_chain_bootstrap_and_suffixes_for_all_reference_types(tmp_path):
    (tmp_path / 'images').mkdir()
    (tmp_path / 'images' / 'Gallery_img_1.jpg').write_bytes(b'original')
    (tmp_path / 'image_rename_map.json').write_text(json.dumps({'a.jpg': 'old.jpg', 'old.jpg': 'Gallery_img_1.jpg'}), encoding='utf-8')
    output = tmp_path / 'response_Chapter.xhtml'
    output.write_text('''<html><body><img src="../images/a.jpg?q=1#part">
        <svg><image xlink:href="images/a.jpg#svg"/></svg>
        <object data="images/a.jpg?object"></object><video poster="images/a.jpg#poster"></video>
        <div style="background-image: url('images/a.jpg?style#fragment')"></div>
        <img src="https://example.com/a.jpg"><img src="data:image/png;base64,aaa"></body></html>''', encoding='utf-8')
    assert tracking.repair_image_references(tmp_path) == 1
    markup = output.read_text(encoding='utf-8')
    for suffix in ('?q=1#part', '#svg', '?object', '#poster', '?style#fragment'):
        assert 'Gallery_img_1.jpg' + suffix in markup
    assert 'https://example.com/a.jpg' in markup
    assert 'data:image/png;base64,aaa' in markup
    assert tracking.repair_image_references(tmp_path) == 0


def test_new_targeted_page_preserves_other_page_bindings(tmp_path):
    inventory(tmp_path, {'OEBPS/Images/a.jpg': b'old'})
    first = page(tmp_path, 'response_First.xhtml', ['a.jpg'])
    tracking.repair_image_references(tmp_path)
    binding = state(tmp_path)['pages'][first.name]
    page(tmp_path, 'response_Second.xhtml', ['a.jpg'])
    tracking.repair_image_references(tmp_path)
    assert state(tmp_path)['pages'][first.name] == binding
    assert len(state(tmp_path)['pages']) == 2


def test_unknown_or_corrupt_sidecar_prevents_destructive_snapshot(tmp_path):
    path = tmp_path / tracking.SIDECAR
    path.write_text('{"version": 99}', encoding='utf-8')
    with pytest.raises(ValueError):
        tracking.snapshot_before_refresh(tmp_path)
    assert path.read_text(encoding='utf-8') == '{"version": 99}'


def test_reference_metadata_is_not_a_progress_chapter():
    from library_core import _is_progress_sidecar_entry
    assert _is_progress_sidecar_entry('image_reference_map.json')


@pytest.mark.parametrize('replace_first', [False, True])
def test_interrupted_html_repair_recovers_bindings_with_overlapping_names(tmp_path, monkeypatch, replace_first):
    inventory(tmp_path, {'OEBPS/Images/a.jpg': b'A', 'OEBPS/Images/b.jpg': b'B'})
    output = page(tmp_path, 'response_Chapter.xhtml', ['a.jpg', 'b.jpg'])
    tracking.snapshot_before_refresh(tmp_path)
    inventory(tmp_path, {'OEBPS/Images/a.jpg': b'A', 'OEBPS/Images/b.jpg': b'B'},
              {'a.jpg': 'b.jpg', 'b.jpg': 'c.jpg'})
    real_replace = tracking.os.replace

    def interrupted(source, destination):
        if Path(destination) == output:
            if replace_first:
                real_replace(source, destination)
            raise OSError('simulated process interruption')
        return real_replace(source, destination)

    monkeypatch.setattr(tracking.os, 'replace', interrupted)
    with pytest.raises(OSError, match='interruption'):
        tracking.repair_image_references(tmp_path)
    assert output.name in state(tmp_path)['repairs']
    monkeypatch.setattr(tracking.os, 'replace', real_replace)
    tracking.repair_image_references(tmp_path)
    assert sources(output) == ['images/b.jpg', 'images/c.jpg']
    assert not state(tmp_path)['repairs']
    assert tracking.repair_image_references(tmp_path) == 0


def test_explicit_retranslation_replaces_unresolved_historical_parent(tmp_path):
    inventory(tmp_path, {'OEBPS/Images/a.jpg': b'old'})
    output = page(tmp_path, 'Chapter.xhtml', ['a.jpg'])
    tracking.snapshot_before_refresh(tmp_path)
    old = state(tmp_path)['pages'][output.name]['a.jpg']
    inventory(tmp_path, {'OEBPS/Images/a.jpg': b'new'})
    assert tracking.repair_image_references(tmp_path) == 0
    page(tmp_path, 'Chapter.xhtml', ['a.jpg'])
    tracking.record_translated_page(tmp_path, output.name)
    assert state(tmp_path)['pages'][output.name]['a.jpg'] != old
    assert tracking.repair_image_references(tmp_path) == 0


def test_alias_cycles_are_not_used_to_guess_a_missing_image(tmp_path):
    (tmp_path / 'images').mkdir()
    (tmp_path / 'images' / 'unrelated.jpg').write_bytes(b'image')
    (tmp_path / 'image_rename_map.json').write_text('{"a.jpg":"b.jpg","b.jpg":"a.jpg"}', encoding='utf-8')
    output = page(tmp_path, 'response_Chapter.xhtml', ['a.jpg'])
    before = output.read_bytes()
    assert tracking.repair_image_references(tmp_path) == 0
    assert output.read_bytes() == before


def test_extraction_rewrites_all_svg_attributes_and_preserves_suffixes():
    import Chapter_Extractor as extractor
    soup = BeautifulSoup('<svg><image href="images/a.jpg?one#two" xlink:href="images/a.jpg#three"/></svg>', 'html.parser')
    assert extractor._update_image_refs_in_soup(soup, {'a.jpg': 'Gallery_img_1.jpg'})
    assert soup.image['href'] == 'images/Gallery_img_1.jpg?one#two'
    assert soup.image['xlink:href'] == 'images/Gallery_img_1.jpg#three'


def test_reference_case_change_does_not_rebind_a_reused_filename(tmp_path):
    inventory(tmp_path, {'OEBPS/Images/a.jpg': b'old'}, {'a.jpg': 'Section5_img_1.jpg'})
    output = page(tmp_path, 'response_Chapter.xhtml', ['Section5_img_1.jpg'])
    tracking.snapshot_before_refresh(tmp_path)
    inventory(tmp_path, {'OEBPS/Images/a.jpg': b'new', 'OEBPS/Images/b.jpg': b'old'},
              {'a.jpg': 'Section5_img_1.jpg', 'b.jpg': 'Section6_img_1.jpg'})
    page(tmp_path, output.name, ['section5_IMG_1.JPG'])
    assert tracking.repair_image_references(tmp_path) == 1
    assert sources(output) == ['images/Section6_img_1.jpg']


@pytest.mark.parametrize('mode', ['pdf', 'passthrough'])
def test_existing_repair_entrypoint_preserves_pdf_and_passthrough(tmp_path, monkeypatch, mode):
    from TransateKRtoEN import retroactive_update_image_references
    import output_workspace
    inventory(tmp_path, {'OEBPS/Images/a.jpg': b'old'}, {'a.jpg': 'Gallery_img_1.jpg'})
    output = page(tmp_path, 'response_Chapter.xhtml', ['a.jpg'])
    before = output.read_bytes()
    if mode == 'pdf':
        monkeypatch.setattr(output_workspace, 'read_workspace_source_path', lambda _folder: 'source.pdf')
        monkeypatch.delenv('IMAGE_MODE_EPUB_PASSTHROUGH', raising=False)
    else:
        monkeypatch.setenv('IMAGE_MODE_EPUB_PASSTHROUGH', '1')
    retroactive_update_image_references(str(tmp_path))
    assert output.read_bytes() == before


def test_progress_completion_rebinds_a_retranslated_page(tmp_path):
    from TransateKRtoEN import ProgressManager
    inventory(tmp_path, {'OEBPS/Images/a.jpg': b'old'})
    output = page(tmp_path, 'response_Chapter.xhtml', ['a.jpg'])
    tracking.snapshot_before_refresh(tmp_path)
    inventory(tmp_path, {'OEBPS/Images/a.jpg': b'new'})
    old_binding = state(tmp_path)['pages'][output.name]['a.jpg']
    manager = ProgressManager(str(tmp_path))
    manager.update(0, 1, 'source-hash', output.name, status='in_progress')
    page(tmp_path, output.name, ['a.jpg'])
    manager.update(0, 1, 'source-hash', output.name, status='completed')
    assert state(tmp_path)['pages'][output.name]['a.jpg'] != old_binding
    assert tracking.repair_image_references(tmp_path) == 0


@pytest.mark.parametrize('retain_names', [False, True])
def test_real_resource_refresh_and_compiler_keep_shared_image_identity(tmp_path, monkeypatch, retain_names):
    import Chapter_Extractor as extractor
    from TransateKRtoEN import retroactive_update_image_references
    from epub_converter import EPUBCompiler
    from PIL import Image

    def png(color):
        buffer = io.BytesIO()
        Image.new('RGB', (2, 2), color).save(buffer, format='PNG')
        return buffer.getvalue()

    original_image, new_image = png('red'), png('blue')
    folder = tmp_path / 'workspace'
    folder.mkdir()
    epub = tmp_path / 'source.epub'

    def extract(updated):
        entries = [('Section0005', 'added.png')] if updated else []
        entries += [('Section0006' if updated else 'Section0005', 'shared.png'),
                    ('Section0009' if updated else 'Section0008', 'shared.png'), ('Gallery', 'shared.png')]
        chapters = []
        with zipfile.ZipFile(epub, 'w') as archive:
            archive.writestr('META-INF/container.xml', '<container xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles><rootfile full-path="OEBPS/content.opf"/></rootfiles></container>')
            manifest = ''.join(f'<item id="c{i}" href="Text/{name}.xhtml" media-type="application/xhtml+xml"/>' for i, (name, _) in enumerate(entries))
            manifest += '<item id="shared" href="Images/shared.png" media-type="image/png"/>'
            if updated:
                manifest += '<item id="added" href="Images/added.png" media-type="image/png"/>'
            spine = ''.join(f'<itemref idref="c{i}"/>' for i in range(len(entries)))
            archive.writestr('OEBPS/content.opf', '<package xmlns="http://www.idpf.org/2007/opf" version="2.0"><metadata xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>Shared Test</dc:title><dc:language>en</dc:language></metadata><manifest>' + manifest + '</manifest><spine>' + spine + '</spine></package>')
            archive.writestr('OEBPS/Images/shared.png', original_image)
            if updated:
                archive.writestr('OEBPS/Images/added.png', new_image)
            for i, (name, image) in enumerate(entries):
                markup = f'<html><head><title>{name}</title></head><body><p>Chapter text</p><img src="../Images/{image}"></body></html>'
                member = f'OEBPS/Text/{name}.xhtml'
                archive.writestr(member, markup)
                chapters.append({'num': i + 1, 'title': name, 'filename': member, 'original_basename': name,
                                 'body': markup, 'has_images': True, 'image_count': 1, 'file_size': len(markup)})
        monkeypatch.setattr(extractor, '_extract_chapters_universal', lambda *_args: (chapters, 'english'))
        with zipfile.ZipFile(epub) as archive:
            return extractor.extract_chapters(archive, str(folder), parser='html.parser', progress_callback=lambda _msg: None)

    monkeypatch.delenv('SINGLE_CHAPTER_FILTER', raising=False)
    monkeypatch.delenv('IMAGE_MODE_EPUB_PASSTHROUGH', raising=False)
    monkeypatch.setenv('EXTRACTION_MODE', 'comprehensive')
    monkeypatch.setenv('EXTRACTION_WORKERS', '1')
    monkeypatch.setenv('DOWNLOAD_REMOTE_IMAGE_URLS', '0')
    first = extract(False)
    assert all('Section0005_img_1.png' in chapter['body'] for chapter in first)
    prefix = '' if retain_names else 'response_'
    outputs = [page(folder, prefix + name + '.xhtml', ['Section0005_img_1.png'])
               for name in ('Gallery', 'Section0005', 'Section0008')]
    second = extract(True)
    assert 'Section0006_img_1.png' in second[1]['body']
    retroactive_update_image_references(str(folder))
    assert all(sources(path) == ['images/Section0006_img_1.png'] for path in outputs)
    assert (folder / 'images' / 'Section0006_img_1.png').read_bytes() == original_image
    compiler = EPUBCompiler(str(folder), log_callback=lambda _msg: None)
    for output in outputs:
        compiled, missing = compiler._process_chapter_images(output.read_text(encoding='utf-8'),
                                                             {'Section0006_img_1.png': 'compiled-shared.png'})
        assert 'images/compiled-shared.png' in compiled
        assert missing == []
    before = (folder / tracking.SIDECAR).read_bytes()
    retroactive_update_image_references(str(folder))
    assert (folder / tracking.SIDECAR).read_bytes() == before
