"""mobile.patch 11. (tensor split across the NPU and the GPU): ggml_backend_meta_buffer_set_tensor stages a
whole-tensor write per device and sends each device's slice with ONE ggml_backend_tensor_set, because OpenCL's
quantised set_tensor aborts on a second partial write. Text-level checks of the patch section (nothing compiles
here): the guard, the single write per device, the partial path forwarding the old calls unchanged, the header.

mobile.patch 12. (tensor split with a dense FFN weight the meta device cannot hold, e.g. IQ4_XS on Hexagon):
load_tensors notes the layers whose ffn_gate/ffn_up landed outside the meta buffer and the split-state callback
mirrors the FFN weights of those layers that are in it. Text-level checks of the two src/ sections.
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
    assert "an eleventh one for tensor split across the NPU and the GPU and a twelfth one for\n" in header
    assert header.index("\n10. Optional NPU backend") < header.index("\n11. Tensor split") < \
        header.index("Upstream's description:")
    assert "#   - patches/mobile.patch 11.: tensor split across the NPU and the GPU" in RECIPE.read_text(encoding="utf-8")
    assert "#   - patches/mobile.patch 12.: tensor split with a dense FFN weight" in RECIPE.read_text(encoding="utf-8")


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


# --- mobile.patch 12. -----------------------------------------------------------------------------------------

MODEL_CPP = "vendor/llama.cpp/src/llama-model.cpp"
MODEL_H = "vendor/llama.cpp/src/llama-model.h"


def _model_section(name: str) -> list[str]:
    return _file_diffs(PATCH.read_text(encoding="utf-8"))[name].splitlines()


def _model_added(name: str) -> str:
    return "\n".join(line[1:] for line in _model_section(name)
                     if line.startswith("+") and not line.startswith("+++")) + "\n"


def test_item_12_is_in_the_header_after_item_11():
    header = PATCH.read_text(encoding="utf-8").split("diff --git", 1)[0]
    assert "\n12. Tensor split with a dense FFN weight the meta device cannot hold (split_mode TENSOR):\n" in header
    assert "tensor split with a weight the meta device cannot hold:\n" in header
    assert header.index("\n11. Tensor split") < header.index("\n12. Tensor split") < \
        header.index("Upstream's description:")


def test_item_12_only_adds_lines_to_the_model_sources():
    names = list(_file_diffs(PATCH.read_text(encoding="utf-8")))
    assert MODEL_CPP in names and MODEL_H in names and names == sorted(names), names
    for name in (MODEL_CPP, MODEL_H):
        removed = [line for line in _model_section(name) if line.startswith("-") and not line.startswith("---")]
        assert removed == [], (name, removed)


def test_the_userdata_carries_the_layer_set():
    added = _model_added(MODEL_H)
    assert "    std::unordered_set<uint32_t> glossarion_ffn_mirrored_layers;\n" in added
    # the struct's two upstream fields stay as context, in front of the new one
    section = _model_section(MODEL_H)
    assert " struct llama_meta_device_get_split_state_userdata {" in section
    assert section.index("     const struct llama_model * model;") < \
        section.index("+    std::unordered_set<uint32_t> glossarion_ffn_mirrored_layers;")


def test_load_tensors_notes_gate_up_outside_the_meta_buffer_before_any_buffer_exists():
    section = _model_section(MODEL_CPP)
    added = _model_added(MODEL_CPP)
    # the set is rebuilt on every load and only filled when a meta device is in use (split mode TENSOR)
    assert "    get_split_state_ud.glossarion_ffn_mirrored_layers.clear();\n" in added
    assert ("    if (std::any_of(devices.begin(), devices.end(), [](const llama_device & d) { return d.is_meta; })) {\n"
            in added)
    # dense FFN tensors only, not the MoE experts / shared experts / router
    assert ('        static const std::regex pattern_ffn_dense("blk\\\\.(\\\\d+)\\\\.ffn_(gate|up|gate_up|down)\\\\.(weight|bias)");\n'
            in added)
    # a context counts as outside when its buffer type does not belong to the meta device
    assert ("            const bool in_meta = buft_dev != nullptr && std::any_of(devices.begin(), devices.end(),\n"
            "                [buft_dev](const llama_device & d) { return d.is_meta && d.dev == buft_dev; });\n") in added
    # inside: any dense FFN tensor marks the layer as present in the meta buffer; outside: only a gate/up weight
    assert ("                if (in_meta) {\n"
            "                    layers_in_meta.insert(il);\n"
            "                } else if (m[2].str() != \"down\" && m[3].str() == \"weight\") {\n"
            "                    gate_up_outside.emplace(il, std::make_pair(cur, ctx_key.buft));\n"
            "                }\n") in added
    # a layer is mirrored only when its gate/up is outside AND some of its FFN is inside (a layer wholly on the
    # CPU is never asked for a split state and gets no warning)
    assert ("        for (const auto & [il, where] : gate_up_outside) {\n"
            "            if (layers_in_meta.count(il) == 0) {\n") in added
    assert "            get_split_state_ud.glossarion_ffn_mirrored_layers.insert(il);\n" in added
    assert added.count("glossarion_ffn_mirrored_layers.insert(") == 1
    # the block runs right before tensors_by_name is filled, i.e. before the buffers (and the split states) exist
    end = max(i for i, line in enumerate(section) if line.startswith("+"))
    assert section[end + 1] == "     // populate tensors_by_name"


def test_the_callback_mirrors_only_the_noted_layers_dense_ffn_weights():
    section = _model_section(MODEL_CPP)
    added = _model_added(MODEL_CPP)
    assert ('    static const std::regex pattern_glossarion_ffn_dense  '
            '("blk\\\\.\\\\d+\\\\.ffn_(gate|up|gate_up|down)\\\\.(weight|bias)");\n') in added
    rule = ("        if (!ud->glossarion_ffn_mirrored_layers.empty() && "
            "std::regex_match(tensor_name, pattern_glossarion_ffn_dense)) {\n"
            "            const tensor_config tc_mirrored = get_tensor_config_impl(GGML_BACKEND_SPLIT_AXIS_MIRRORED);\n"
            "            if (ud->glossarion_ffn_mirrored_layers.count(tc_mirrored.il) > 0) {\n"
            "                return tc_mirrored;\n"
            "            }\n"
            "        }\n")
    assert rule in added
    # the rule sits right in front of upstream's FFN rules, so every other layer still gets its axis-1/axis-0 split
    i = section.index("+        if (!ud->glossarion_ffn_mirrored_layers.empty() && "
                      "std::regex_match(tensor_name, pattern_glossarion_ffn_dense)) {")
    after = [line for line in section[i:] if not line.startswith("+")]
    assert after[:2] == ["         // FFN",
                         "         if (std::regex_match(tensor_name, pattern_ffn_up_weight) || "
                         "std::regex_match(tensor_name, pattern_ffn_gate_weight)) {"], after[:2]
