"""Review glossary names in a workspace and retain the findings with the job."""

from dataclasses import asdict


ICON = "FIND_IN_PAGE"


def run(ctx):
    import glossary_name_check as checker

    params = ctx.params
    folders = ctx.inputs
    output_file = params.get("output_file")
    ctx.phase("Checking glossary names")
    downloading = False

    def on_download(report):
        nonlocal downloading
        if not downloading:
            ctx.phase("Downloading NER model")
            downloading = True
        ctx.host.emit("progress", label="Downloading NER model",
                      completed=int(100 * report.fraction) if report.fraction is not None else 0,
                      total=100 if report.fraction is not None else 0)

    ner = checker.resolve_ner(bool(params.get("ner")), ctx.log, progress=on_download,
                              cancel=ctx.stop_requested, retry=True)
    workers = checker.chapter_workers(ner, chosen=int(params.get("workers") or 0))
    reviews = []
    for index, folder in enumerate(folders, 1):
        if ctx.stop_requested():
            return
        label = f"Checking folder {index} of {len(folders)}" if len(folders) > 1 else "Checking glossary names"
        ctx.phase(label)
        ctx.host.emit("progress", label=label, completed=0, total=0)

        def on_chapter(done, total):
            ctx.host.emit("progress", label=label, completed=done + 1, total=total)

        result = checker.review_folder(folder, params["scope"], stop=ctx.stop_requested, ner=ner,
                                       progress=on_chapter, workers=workers, settings=params.get("layers"),
                                       selected_files=[output_file] if output_file else None,
                                       progress_path=params.get("progress_path"))
        reviews.append(asdict(result))
    if not ctx.stop_requested():
        ctx.set_result(glossary_name_reviews=reviews)
        ctx.phase("Glossary names checked")


def load_reviews(records):
    from glossary_name_check import ChapterResult, Finding, FolderReview

    return [FolderReview(folder=row["folder"], glossary_path=row["glossary_path"], problem=row["problem"],
                         chapters=[ChapterResult(chapter["output_file"],
                                                 [Finding(**finding) for finding in chapter["findings"]])
                                   for chapter in row["chapters"]]) for row in records]


KINDS = {"glossary_names": {"verb": "Checking glossary names", "icon": ICON, "stop_kind": "translation", "run": run}}
