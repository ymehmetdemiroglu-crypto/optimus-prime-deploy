
import os
import langgraph

package_dir = os.path.dirname(langgraph.__file__)
print(f"Searching in: {package_dir}")

for root, dirs, files in os.walk(package_dir):
    for file in files:
        if file.endswith(".py"):
            filepath = os.path.join(root, file)
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                if "AgentExecutor" in content:
                    print(f"Found AgentExecutor usage in: {filepath}")
