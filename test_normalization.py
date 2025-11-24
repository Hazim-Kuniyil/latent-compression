"""Quick test to verify HotpotQA normalization works correctly."""
import sys
sys.path.insert(0, 'src')

from data_hotpot import _normalize_hotpot_fields

# Test 1: HF format (lists)
print("Test 1: HuggingFace format (lists)")
hf_supporting_facts = [["Title1", 0], ["Title2", 1]]
hf_context = [
    ["Title1", ["Sentence 1.", "Sentence 2."]],
    ["Title2", ["Sentence 3.", "Sentence 4."]]
]

sf_norm, ctx_norm = _normalize_hotpot_fields(hf_supporting_facts, hf_context)
print("Supporting facts:", sf_norm)
print("Context:", ctx_norm)
assert sf_norm == {"title": ["Title1", "Title2"], "sent_id": [0, 1]}
assert ctx_norm == {
    "title": ["Title1", "Title2"],
    "sentences": [["Sentence 1.", "Sentence 2."], ["Sentence 3.", "Sentence 4."]]
}
print("✓ HF format test passed\n")

# Test 2: Already normalized dict format
print("Test 2: Already normalized dict format")
dict_supporting_facts = {"title": ["TitleA"], "sent_id": [0]}
dict_context = {"title": ["TitleA"], "sentences": [["Sentence A."]]}

sf_norm2, ctx_norm2 = _normalize_hotpot_fields(dict_supporting_facts, dict_context)
print("Supporting facts:", sf_norm2)
print("Context:", ctx_norm2)
assert sf_norm2 == dict_supporting_facts
assert ctx_norm2 == dict_context
print("✓ Dict format test passed\n")

# Test 3: Empty/malformed data
print("Test 3: Empty data")
sf_norm3, ctx_norm3 = _normalize_hotpot_fields([], [])
print("Supporting facts:", sf_norm3)
print("Context:", ctx_norm3)
assert sf_norm3 == {"title": [], "sent_id": []}
assert ctx_norm3 == {"title": [], "sentences": []}
print("✓ Empty data test passed\n")

print("All normalization tests passed!")
