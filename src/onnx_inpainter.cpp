/**
 * ONNX Runtime C++ Inference Module for Inpainting
 * Lightweight native implementation for high-performance inference
 */

// MinGW compatibility - define Microsoft SAL annotations as empty
#if defined(__MINGW32__) || defined(__MINGW64__)
    #ifndef _Frees_ptr_opt_
        #define _Frees_ptr_opt_
    #endif
    #ifndef _Ret_maybenull_
        #define _Ret_maybenull_
    #endif
    #ifndef _In_opt_
        #define _In_opt_
    #endif
    #ifndef _Inout_
        #define _Inout_
    #endif
    #ifndef _Out_
        #define _Out_
    #endif
    #ifndef _In_
        #define _In_
    #endif
#endif

#include <onnxruntime_c_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef _WIN32
    #include <windows.h>
#endif

#ifdef _WIN32
    #define EXPORT extern "C" __declspec(dllexport)
#else
    #define EXPORT extern "C" __attribute__((visibility("default")))
#endif

// ONNX Runtime API
const OrtApi* g_ort = NULL;

// Session handle
typedef struct {
    OrtSession* session;
    OrtAllocator* allocator;
    char** input_names;
    char** output_names;
    size_t num_inputs;
    size_t num_outputs;
    int64_t* input_shape;
    bool is_dynamic;
} ONNXInpainter;

// Error checking macro - returns -1 on error
#define CHECK_STATUS_INT(expr)                                           \
  do {                                                                   \
    OrtStatus* status = (expr);                                          \
    if (status != NULL) {                                                \
      const char* msg = g_ort->GetErrorMessage(status);                  \
      fprintf(stderr, "ONNX Runtime Error: %s\n", msg);                  \
      g_ort->ReleaseStatus(status);                                      \
      return -1;                                                         \
    }                                                                    \
  } while (0)

// Error checking macro - returns NULL on error
#define CHECK_STATUS(expr)                                               \
  do {                                                                   \
    OrtStatus* status = (expr);                                          \
    if (status != NULL) {                                                \
      const char* msg = g_ort->GetErrorMessage(status);                  \
      fprintf(stderr, "ONNX Runtime Error: %s\n", msg);                  \
      g_ort->ReleaseStatus(status);                                      \
      return NULL;                                                       \
    }                                                                    \
  } while (0)

// Error checking macro for goto cleanup
#define CHECK_STATUS_GOTO(expr)                                          \
  do {                                                                   \
    OrtStatus* status = (expr);                                          \
    if (status != NULL) {                                                \
      const char* msg = g_ort->GetErrorMessage(status);                  \
      fprintf(stderr, "ONNX Runtime Error: %s\n", msg);                  \
      g_ort->ReleaseStatus(status);                                      \
      goto cleanup;                                                      \
    }                                                                    \
  } while (0)

// Initialize ONNX Runtime
EXPORT int onnx_init() {
    g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    if (!g_ort) {
        fprintf(stderr, "Failed to initialize ONNX Runtime API\n");
        return -1;
    }
    return 0;
}

// Create session with GPU support if available
EXPORT ONNXInpainter* onnx_create_session(const char* model_path, int use_gpu) {
    if (onnx_init() != 0) {
        return NULL;
    }

    ONNXInpainter* inpainter = (ONNXInpainter*)malloc(sizeof(ONNXInpainter));
    if (!inpainter) return NULL;

    memset(inpainter, 0, sizeof(ONNXInpainter));

    // Create environment
    OrtEnv* env = NULL;
    CHECK_STATUS(g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "InpaintModel", &env));

    // Session options
    OrtSessionOptions* session_options = NULL;
    CHECK_STATUS(g_ort->CreateSessionOptions(&session_options));
    
    // Optimize for inference
    CHECK_STATUS(g_ort->SetSessionGraphOptimizationLevel(session_options, ORT_ENABLE_EXTENDED));
    
    // Disable memory arena for lower memory footprint
    CHECK_STATUS(g_ort->DisableMemPattern(session_options));
    CHECK_STATUS(g_ort->DisableCpuMemArena(session_options));

    // GPU support
    if (use_gpu) {
        OrtCUDAProviderOptions cuda_options;
        memset(&cuda_options, 0, sizeof(cuda_options));
        cuda_options.device_id = 0;
        
        OrtStatus* cuda_status = g_ort->SessionOptionsAppendExecutionProvider_CUDA(
            session_options, &cuda_options);
        
        if (cuda_status != NULL) {
            fprintf(stderr, "CUDA not available, using CPU\n");
            g_ort->ReleaseStatus(cuda_status);
        }
    }

    // Create session
#ifdef _WIN32
    // Windows uses wide strings - convert UTF-8 to wide
    int len = MultiByteToWideChar(CP_UTF8, 0, model_path, -1, NULL, 0);
    if (len == 0) {
        fprintf(stderr, "Failed to convert model path to wide string\n");
        free(inpainter);
        return NULL;
    }
    wchar_t* wpath = (wchar_t*)malloc(len * sizeof(wchar_t));
    if (!wpath) {
        free(inpainter);
        return NULL;
    }
    MultiByteToWideChar(CP_UTF8, 0, model_path, -1, wpath, len);
    OrtStatus* create_status = g_ort->CreateSession(env, wpath, session_options, &inpainter->session);
    free(wpath);
    if (create_status != NULL) {
        const char* msg = g_ort->GetErrorMessage(create_status);
        fprintf(stderr, "ONNX Runtime Error: %s\n", msg);
        g_ort->ReleaseStatus(create_status);
        free(inpainter);
        return NULL;
    }
#else
    CHECK_STATUS(g_ort->CreateSession(env, model_path, session_options, &inpainter->session));
#endif

    // Get allocator
    CHECK_STATUS(g_ort->GetAllocatorWithDefaultOptions(&inpainter->allocator));

    // Get input/output info
    CHECK_STATUS(g_ort->SessionGetInputCount(inpainter->session, &inpainter->num_inputs));
    CHECK_STATUS(g_ort->SessionGetOutputCount(inpainter->session, &inpainter->num_outputs));

    // Allocate name arrays
    inpainter->input_names = (char**)malloc(sizeof(char*) * inpainter->num_inputs);
    inpainter->output_names = (char**)malloc(sizeof(char*) * inpainter->num_outputs);

    // Get input names and shapes
    for (size_t i = 0; i < inpainter->num_inputs; i++) {
        CHECK_STATUS(g_ort->SessionGetInputName(inpainter->session, i, 
                                                 inpainter->allocator, 
                                                 &inpainter->input_names[i]));
        
        // Get input shape for first input (image)
        if (i == 0) {
            OrtTypeInfo* type_info = NULL;
            CHECK_STATUS(g_ort->SessionGetInputTypeInfo(inpainter->session, i, &type_info));
            
            const OrtTensorTypeAndShapeInfo* tensor_info = NULL;
            CHECK_STATUS(g_ort->CastTypeInfoToTensorInfo(type_info, &tensor_info));
            
            size_t num_dims = 0;
            CHECK_STATUS(g_ort->GetDimensionsCount(tensor_info, &num_dims));
            
            inpainter->input_shape = (int64_t*)malloc(sizeof(int64_t) * num_dims);
            CHECK_STATUS(g_ort->GetDimensions(tensor_info, inpainter->input_shape, num_dims));
            
            // Check if dynamic (any dimension is -1 or 0)
            inpainter->is_dynamic = false;
            for (size_t d = 0; d < num_dims; d++) {
                if (inpainter->input_shape[d] <= 0) {
                    inpainter->is_dynamic = true;
                    break;
                }
            }
            
            g_ort->ReleaseTypeInfo(type_info);
        }
    }

    // Get output names
    for (size_t i = 0; i < inpainter->num_outputs; i++) {
        CHECK_STATUS(g_ort->SessionGetOutputName(inpainter->session, i, 
                                                  inpainter->allocator, 
                                                  &inpainter->output_names[i]));
    }

    g_ort->ReleaseSessionOptions(session_options);

    return inpainter;
}

// Run inference
EXPORT int onnx_infer(ONNXInpainter* inpainter,
                     const float* image_data,
                     const float* mask_data,
                     int batch,
                     int channels,
                     int height,
                     int width,
                     float* output_data) {
    if (!inpainter || !inpainter->session) {
        fprintf(stderr, "Invalid inpainter session\n");
        return -1;
    }

    // Create input tensors
    OrtMemoryInfo* memory_info = NULL;
    CHECK_STATUS_INT(g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info));

    // Image tensor shape [batch, channels, height, width]
    int64_t image_shape[] = {batch, channels, height, width};
    size_t image_size = batch * channels * height * width;

    OrtValue* image_tensor = NULL;
    CHECK_STATUS_INT(g_ort->CreateTensorWithDataAsOrtValue(
        memory_info,
        (void*)image_data,
        image_size * sizeof(float),
        image_shape,
        4,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        &image_tensor));

    // Mask tensor shape [batch, 1, height, width]
    int64_t mask_shape[] = {batch, 1, height, width};
    size_t mask_size = batch * height * width;

    OrtValue* mask_tensor = NULL;
    CHECK_STATUS_INT(g_ort->CreateTensorWithDataAsOrtValue(
        memory_info,
        (void*)mask_data,
        mask_size * sizeof(float),
        mask_shape,
        4,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        &mask_tensor));

    // Prepare input/output
    const OrtValue* input_tensors[] = {image_tensor, mask_tensor};
    OrtValue* output_tensor = NULL;

    // Run inference
    OrtStatus* status = g_ort->Run(
        inpainter->session,
        NULL,  // run options
        (const char* const*)inpainter->input_names,
        input_tensors,
        inpainter->num_inputs,
        (const char* const*)inpainter->output_names,
        inpainter->num_outputs,
        &output_tensor);

    if (status != NULL) {
        const char* msg = g_ort->GetErrorMessage(status);
        fprintf(stderr, "Inference failed: %s\n", msg);
        g_ort->ReleaseStatus(status);
        g_ort->ReleaseValue(image_tensor);
        g_ort->ReleaseValue(mask_tensor);
        g_ort->ReleaseMemoryInfo(memory_info);
        return -1;
    }

    // Copy output data
    float* output_buffer = NULL;
    OrtStatus* get_status = g_ort->GetTensorMutableData(output_tensor, (void**)&output_buffer);
    if (get_status != NULL) {
        const char* msg = g_ort->GetErrorMessage(get_status);
        fprintf(stderr, "Failed to get tensor data: %s\n", msg);
        g_ort->ReleaseStatus(get_status);
        g_ort->ReleaseValue(output_tensor);
        g_ort->ReleaseValue(image_tensor);
        g_ort->ReleaseValue(mask_tensor);
        g_ort->ReleaseMemoryInfo(memory_info);
        return -1;
    }
    memcpy(output_data, output_buffer, image_size * sizeof(float));

    // Cleanup
    g_ort->ReleaseValue(output_tensor);
    g_ort->ReleaseValue(image_tensor);
    g_ort->ReleaseValue(mask_tensor);
    g_ort->ReleaseMemoryInfo(memory_info);

    return 0;
}

// Get model info
EXPORT int onnx_get_input_shape(ONNXInpainter* inpainter, int* shape_out) {
    if (!inpainter || !inpainter->input_shape) return -1;
    
    // Return [batch, channels, height, width]
    shape_out[0] = (int)inpainter->input_shape[0];
    shape_out[1] = (int)inpainter->input_shape[1];
    shape_out[2] = (int)inpainter->input_shape[2];
    shape_out[3] = (int)inpainter->input_shape[3];
    
    return inpainter->is_dynamic ? 1 : 0;
}

// Cleanup
EXPORT void onnx_destroy_session(ONNXInpainter* inpainter) {
    if (!inpainter) return;

    if (inpainter->session) {
        g_ort->ReleaseSession(inpainter->session);
    }

    if (inpainter->input_names) {
        for (size_t i = 0; i < inpainter->num_inputs; i++) {
            if (inpainter->input_names[i]) {
                inpainter->allocator->Free(inpainter->allocator, inpainter->input_names[i]);
            }
        }
        free(inpainter->input_names);
    }

    if (inpainter->output_names) {
        for (size_t i = 0; i < inpainter->num_outputs; i++) {
            if (inpainter->output_names[i]) {
                inpainter->allocator->Free(inpainter->allocator, inpainter->output_names[i]);
            }
        }
        free(inpainter->output_names);
    }

    if (inpainter->input_shape) {
        free(inpainter->input_shape);
    }

    free(inpainter);
}

// Version info
EXPORT const char* onnx_version() {
    if (onnx_init() != 0) return "unknown";
    // GetVersionString may not be available in all versions
    return "1.20.1";
}

// ============================================================================
// RT-DETR Bubble Detection Support
// ============================================================================

// Detection result structure
typedef struct {
    float x1, y1, x2, y2;  // Bounding box coordinates
    float score;            // Confidence score
    int label;              // Class label (0=bubble, 1=text_bubble, 2=text_free)
} Detection;

// Run RT-DETR detection
EXPORT int onnx_detect_bubbles(
    ONNXInpainter* session,
    const float* image_data,   // RGB image data [H, W, 3]
    int height,
    int width,
    int orig_width,            // Original image size before resizing
    int orig_height,
    Detection* detections,     // Output array for detections
    int max_detections,        // Maximum number of detections
    int* num_detections        // Actual number of detections returned
) {
    if (!session || !session->session) {
        fprintf(stderr, "Invalid session\n");
        return -1;
    }

    OrtMemoryInfo* memory_info = NULL;
    OrtStatus* status = g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info);
    if (status != NULL) {
        const char* msg = g_ort->GetErrorMessage(status);
        fprintf(stderr, "Failed to create memory info: %s\n", msg);
        g_ort->ReleaseStatus(status);
        return -1;
    }

    // Image tensor [1, 3, 640, 640]
    int64_t image_shape[] = {1, 3, 640, 640};
    size_t image_size = 1 * 3 * 640 * 640;

    OrtValue* image_tensor = NULL;
    status = g_ort->CreateTensorWithDataAsOrtValue(
        memory_info,
        (void*)image_data,
        image_size * sizeof(float),
        image_shape,
        4,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        &image_tensor);
    
    if (status != NULL) {
        const char* msg = g_ort->GetErrorMessage(status);
        fprintf(stderr, "Failed to create image tensor: %s\n", msg);
        g_ort->ReleaseStatus(status);
        g_ort->ReleaseMemoryInfo(memory_info);
        return -1;
    }

    // Original size tensor [1, 2] - width, height
    int64_t orig_size_data[] = {(int64_t)orig_width, (int64_t)orig_height};
    int64_t size_shape[] = {1, 2};

    OrtValue* size_tensor = NULL;
    status = g_ort->CreateTensorWithDataAsOrtValue(
        memory_info,
        (void*)orig_size_data,
        2 * sizeof(int64_t),
        size_shape,
        2,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
        &size_tensor);
    
    if (status != NULL) {
        const char* msg = g_ort->GetErrorMessage(status);
        fprintf(stderr, "Failed to create size tensor: %s\n", msg);
        g_ort->ReleaseStatus(status);
        g_ort->ReleaseValue(image_tensor);
        g_ort->ReleaseMemoryInfo(memory_info);
        return -1;
    }

    // Prepare inputs
    const char* input_names[] = {"images", "orig_target_sizes"};
    const OrtValue* inputs[] = {image_tensor, size_tensor};
    
    // Output names for RT-DETR
    const char* output_names[] = {"labels", "boxes", "scores"};
    OrtValue* outputs[3] = {NULL, NULL, NULL};

    // Run inference
    status = g_ort->Run(
        session->session,
        NULL,
        input_names,
        inputs,
        2,
        output_names,
        3,
        outputs);

    if (status != NULL) {
        const char* msg = g_ort->GetErrorMessage(status);
        fprintf(stderr, "Inference failed: %s\n", msg);
        g_ort->ReleaseStatus(status);
        g_ort->ReleaseValue(image_tensor);
        g_ort->ReleaseValue(size_tensor);
        g_ort->ReleaseMemoryInfo(memory_info);
        return -1;
    }

    // Declare variables before goto cleanup
    int64_t* labels_data = NULL;
    float* boxes_data = NULL;
    float* scores_data = NULL;
    OrtTensorTypeAndShapeInfo* shape_info = NULL;
    size_t num_dims = 0;
    int64_t* dims = NULL;
    size_t num_det = 1;
    size_t count = 0;
    int result = 0;
    
    // Extract output data
    status = g_ort->GetTensorMutableData(outputs[0], (void**)&labels_data);
    if (status != NULL) {
        g_ort->ReleaseStatus(status);
        result = -1;
        goto cleanup;
    }

    status = g_ort->GetTensorMutableData(outputs[1], (void**)&boxes_data);
    if (status != NULL) {
        g_ort->ReleaseStatus(status);
        result = -1;
        goto cleanup;
    }

    status = g_ort->GetTensorMutableData(outputs[2], (void**)&scores_data);
    if (status != NULL) {
        g_ort->ReleaseStatus(status);
        result = -1;
        goto cleanup;
    }

    // Get number of detections
    status = g_ort->GetTensorTypeAndShape(outputs[0], &shape_info);
    if (status != NULL) {
        g_ort->ReleaseStatus(status);
        result = -1;
        goto cleanup;
    }

    g_ort->GetDimensionsCount(shape_info, &num_dims);
    
    dims = (int64_t*)malloc(sizeof(int64_t) * num_dims);
    g_ort->GetDimensions(shape_info, dims, num_dims);
    
    for (size_t i = 0; i < num_dims; i++) {
        num_det *= dims[i];
    }
    free(dims);
    dims = NULL;
    g_ort->ReleaseTensorTypeAndShapeInfo(shape_info);
    shape_info = NULL;

    // Copy detections to output array
    count = num_det < (size_t)max_detections ? num_det : (size_t)max_detections;
    for (size_t i = 0; i < count; i++) {
        detections[i].x1 = boxes_data[i * 4 + 0];
        detections[i].y1 = boxes_data[i * 4 + 1];
        detections[i].x2 = boxes_data[i * 4 + 2];
        detections[i].y2 = boxes_data[i * 4 + 3];
        detections[i].score = scores_data[i];
        detections[i].label = (int)labels_data[i];
    }
    *num_detections = (int)count;

cleanup:
    // Release tensors
    for (int i = 0; i < 3; i++) {
        if (outputs[i]) {
            g_ort->ReleaseValue(outputs[i]);
        }
    }
    if (image_tensor) g_ort->ReleaseValue(image_tensor);
    if (size_tensor) g_ort->ReleaseValue(size_tensor);
    if (memory_info) g_ort->ReleaseMemoryInfo(memory_info);
    if (dims) free(dims);
    if (shape_info) g_ort->ReleaseTensorTypeAndShapeInfo(shape_info);

    return result;
}
