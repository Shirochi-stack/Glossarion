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
