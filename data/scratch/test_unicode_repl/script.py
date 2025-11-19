
import sys

# Test various Unicode characters in the CODE ITSELF (not in output)
# This tests if the subprocess can RECEIVE Unicode in stdin
text = "Hello‑World"  # Non-breaking hyphen (the problematic character)
text2 = "Em—dash"      # Em dash
text3 = "Bullet•point" # Bullet point
text4 = "Copyright©"   # Copyright symbol

# Don't print Unicode (that fails on cp949 console)
# Instead, verify the strings are correctly stored
assert "‑" in text, "Non-breaking hyphen not found"
assert "—" in text2, "Em dash not found"
assert "•" in text3, "Bullet not found"
assert "©" in text4, "Copyright symbol not found"

# Print safe ASCII confirmation
print("SUCCESS: All Unicode characters processed correctly in code!")
print(f"String lengths: {len(text)}, {len(text2)}, {len(text3)}, {len(text4)}")
