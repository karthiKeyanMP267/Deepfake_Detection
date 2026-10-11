import re

with open("run_final_project_validation.py", "r", encoding="utf-8") as f:
    text = f.read()

# Replace any open(X, "w") or open(X, 'w') with open(X, "w", encoding="utf-8")
text = re.sub(r'open\(([^,]+),\s*"w"\)', r'open(\1, "w", encoding="utf-8")', text)
text = re.sub(r"open\(([^,]+),\s*'w'\)", r'open(\1, "w", encoding="utf-8")', text)

with open("run_final_project_validation.py", "w", encoding="utf-8") as f:
    f.write(text)

print("Successfully replaced all open write calls with encoding='utf-8'!")
