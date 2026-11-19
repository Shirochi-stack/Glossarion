"""mobile.patch 11. (tensor split across the NPU and the GPU): ggml_backend_meta_buffer_set_tensor stages a
whole-tensor write per device and sends each device's slice with ONE ggml_backend_tensor_set, because OpenCL's
quantised set_tensor aborts on a second partial write. Text-level checks of the patch section (nothing compiles
here): the guard, the single write per device, the partial path forwarding the old calls unchanged, the header.
"""

from __future__ import annotations

import re

from test_engine_npu_build import PATCH, RECIPE, _file_diffs

META = "vendor/llama.cpp/ggml/src/ggml-backend-meta.cpp"


def _section() -> str:
    return _file_diffs(PATCH.read_text(encoding="utf-8"))[META]


def _lines(prefix: str) -> list[str]:
    return [line[1:] for line in _section().splitlines()
            if line.startswith(prefix) and not line.startswith(prefix * 3)]


def _function(added: str, name: str) -> str:
    """The body of an added static function, from its signature to the first closing brace at column 0."""
    start = added.index(f"static void {name}(")
    return added[start:added.index("\n}\n", start) + 3]


def test_patch_has_the_meta_section_in_file_order_and_touches_no_other_backend():
    names = list(_file_diffs(PATCH.read_text(encoding="utf-8")))
    assert META in names and names == sorted(names), names
    for other in ("ggml-backend.cpp", "ggml-opencl.cpp"):
        assert not any(n.endswith("/" + other) for n in names), names


def test_header_and_recipe_list_the_item():
    header = PATCH.read_text(encoding="utf-8").split("diff --git", 1)[0]
    assert "\n11. Tensor split across the NPU and the GPU (split_mode TENSOR over HTP0 + GPUOpenCL):\n" in header
    assert "an eleventh one for tensor split across the NPU and the GPU:" in header
    assert header.index("\n10. Optional NPU backend") < header.index("\n11. Tensor split") < \
        header.index("Upstream's description:")
    assert "#   - patches/mobile.patch 11.: tensor split across the NPU and the GPU" in RECIPE.read_text(encoding="utf-8")


def test_the_change_is_marked_with_its_reason():
    added = "\n".join(_lines("+"))
    assert ("// Glossarion: a whole-tensor write of a split weight is staged per device and sent with ONE set_tensor each:\n"
            "// OpenCL's quantised set_tensor converts the whole slice on its first call and aborts on a second partial "
            "write.") in added


def test_staging_is_guarded_by_the_whole_tensor_condition():
    added = _lines("+")
    assert "    const bool whole = offset == 0 && size == ggml_nbytes(tensor);" in added
    text = "\n".join(added) + "\n"
    # a partial write gets no stage; the stages are allocated on first use, so mirrored/partial calls pay nothing
    assert ("    auto stage = [&](size_t j) -> glossarion_meta_stage * {\n"
            "        if (!whole) {\n"
            "            return nullptr;\n"
            "        }\n"
            "        if (stages.empty()) {\n"
            "            stages.resize(n_bufs);\n"
            "        }\n"
            "        return &stages[j];\n"
            "    };\n") in text
    # the single-chunk branch stages only when it would otherwise write several chunks
    assert "                glossarion_meta_stage * stage_j = i_stop - i_start > 1 ? stage(j) : nullptr;" in added


def test_partial_writes_forward_the_old_calls_unchanged():
    removed, added = _lines("-"), _lines("+")
    # every removed line is an old ggml_backend_tensor_set_2d call, re-added with only the name changed and the
    # stage pointer in front: same tensor, source, destination offset, size, count and strides
    assert len(removed) == 3 and all("ggml_backend_tensor_set_2d(simple_tensor, " in r for r in removed), removed
    for old in removed:
        indent, rest = old.split("ggml_backend_tensor_set_2d(simple_tensor, ", 1)
        assert (f"{indent}glossarion_meta_set_2d(stage(j), simple_tensor, {rest}" in added
                or f"{indent}glossarion_meta_set_2d(stage_j, simple_tensor, {rest}" in added), old
    # the continuation lines of the two multi-line calls stay as they were
    context = [line[1:] for line in _section().splitlines() if line.startswith(" ")]
    for line in ("                            simple_offsets[j] + row_start * simple_tensor->nb[1], nbytes,",
                 "                            row_count, simple_tensor->nb[1], tensor->nb[1]);",
                 "                        simple_offsets[j] + row_start * simple_tensor->nb[2], nbytes,",
                 "                        row_count, simple_tensor->nb[2], tensor->nb[2]);"):
        assert line in context, line
    # without a stage the helper is exactly the old call
    helper = _function("\n".join(added) + "\n", "glossarion_meta_set_2d")
    assert ("    if (stage == nullptr) {\n"
            "        ggml_backend_tensor_set_2d(tensor, data, offset, size, n_copies, stride_tensor, stride_data);\n"
            "        return;\n"
            "    }\n") in helper


def test_the_staging_path_copies_like_set_2d_and_writes_no_device():
    helper = _function("\n".join(_lines("+")) + "\n", "glossarion_meta_set_2d")
    staged = helper.split("        return;\n    }\n", 1)[1]
    assert "ggml_backend_tensor_set" not in re.sub(r"//[^\n]*", "", staged)    # comments may name it
    # ggml_backend_tensor_set_2d's semantics: copy i goes from data + i*stride_data to offset + i*stride_tensor
    # and its checks in its order: tensor and buffer before the size-0 return, the data pointer after it
    assert staged.index('"tensor buffer not set"') < staged.index("    if (size == 0) {") < \
        staged.index('GGML_ASSERT(tensor->data != NULL && "tensor not allocated");')
    assert "    for (size_t i = 0; i < n_copies; i++) {" in staged
    assert ("memcpy(stage->bytes.data() + offset + i*stride_tensor, (const char *) data + i*stride_data, size);"
            in staged)
    assert "offset + i*stride_tensor + size <= stage->bytes.size()" in staged
    assert "stage->bytes.resize(ggml_nbytes(tensor));" in staged and "stage->written += n_copies*size;" in staged


def test_exactly_one_whole_write_per_device_in_the_staging_path():
    added = _lines("+")
    text = "\n".join(added) + "\n"
    # the only device write added is the flush's single whole-slice set, after the coverage check
    writes = [line for line in added if re.search(r"ggml_backend_tensor_set(_2d)?\(", re.sub(r"//.*", "", line))]
    assert writes == ["        ggml_backend_tensor_set_2d(tensor, data, offset, size, n_copies, stride_tensor, stride_data);",
                      "    ggml_backend_tensor_set(tensor, stage->bytes.data(), 0, nbytes);"], writes
    flush = _function(text, "glossarion_meta_flush")
    assert flush.index("GGML_ASSERT(stage->written == nbytes && stage->bytes.size() == nbytes);") < \
        flush.index("ggml_backend_tensor_set(tensor, stage->bytes.data(), 0, nbytes);")
    assert "if (stage == nullptr || nbytes == 0) {" in flush            # empty slices are skipped
    # the multi-segment branches flush every device once, right before their return
    assert ("    auto flush_all = [&]() {\n"
            "        for (size_t j = 0; j < n_bufs; j++) {\n"
            "            glossarion_meta_flush(stage(j), ggml_backend_meta_buffer_simple_tensor(tensor, j));\n"
            "        }\n"
            "    };") in text
    section = _section().splitlines()
    calls = [i for i, line in enumerate(section) if line[:1] == "+" and line[1:].strip() == "flush_all();"]
    assert len(calls) == 2 and all(section[i + 1][1:].strip() == "return;" for i in calls), calls
    # the single-chunk branch flushes device j right after its one set_2d
    i = section.index("+                glossarion_meta_flush(stage_j, simple_tensor);")
    assert section[i - 1].startswith("+                glossarion_meta_set_2d(stage_j, simple_tensor, ")
