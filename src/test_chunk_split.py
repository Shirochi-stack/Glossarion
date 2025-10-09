"""
Test script to debug chunk splitting logic
"""
import os
os.environ['DEBUG_CHUNK_SPLITTING'] = '1'

from chapter_splitter import ChapterSplitter

# Simulate the ACTUAL user scenario
# MAX_OUTPUT_TOKENS = 32000
# safety_margin = 500
# compression_factor = 1.0 (user's actual setting)
# available_tokens = (32000 - 500) / 1.0 = 31,500

max_output_tokens = 32000
safety_margin = 500
compression_factor = 1.0  # User's actual compression factor
available_tokens = int((max_output_tokens - safety_margin) / compression_factor)

print("=" * 60)
print("CHUNK SPLITTING TEST")
print("=" * 60)
print(f"Max output tokens: {max_output_tokens:,}")
print(f"Safety margin: {safety_margin:,}")
print(f"Compression factor: {compression_factor}")
print(f"Calculated available_tokens: {available_tokens:,}")
print("=" * 60)

# Create a test chapter with approximately 32,523 tokens
# We need to match the user's exact scenario
test_content = "<body>\n"
test_content += "<h1>Test Chapter 4</h1>\n"
# Create paragraphs that add up to roughly 32,523 tokens
# Each repetition of the lorem text is ~14 tokens
# Target: 32,523 tokens, so we need about 2,323 repetitions
num_paragraphs = 361
reps_per_para = 10  # Roughly 140 tokens per paragraph, so 361 * 90 = 32,490 tokens
for i in range(num_paragraphs):
    test_content += f"<p>This is paragraph {i+1}. " + ("Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * reps_per_para) + "</p>\n"
test_content += "</body>"

print(f"\nTest content created with {num_paragraphs} paragraphs")

# Initialize splitter and test
splitter = ChapterSplitter(model_name="gpt-4o")
print(f"\nCounting tokens in test content...")
actual_tokens = splitter.count_tokens(test_content)
print(f"Actual token count: {actual_tokens:,}")

print(f"\nSplitting with max_tokens={available_tokens:,}...")
chunks = splitter.split_chapter(test_content, available_tokens)

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
print(f"Number of chunks created: {len(chunks)}")
print(f"Expected chunks (theoretical): {(actual_tokens / available_tokens):.2f}")
print(f"Expected chunks (rounded up): {-(-actual_tokens // available_tokens)}")  # Ceiling division

# Verify each chunk
print("\nChunk verification:")
total_verified = 0
for chunk_html, chunk_idx, total_chunks in chunks:
    chunk_tokens = splitter.count_tokens(chunk_html)
    total_verified += chunk_tokens
    status = "✓" if chunk_tokens <= available_tokens else "✗ OVERFLOW"
    print(f"  Chunk {chunk_idx}/{total_chunks}: {chunk_tokens:,} tokens {status}")

print(f"\nTotal tokens (sum of chunks): {total_verified:,}")
print(f"Original tokens: {actual_tokens:,}")
print(f"Difference: {total_verified - actual_tokens:,} tokens")

if len(chunks) > 2:
    print(f"\n⚠️ WARNING: Created {len(chunks)} chunks when {-(-actual_tokens // available_tokens)} expected!")
else:
    print(f"\n✓ Chunk count looks reasonable")
