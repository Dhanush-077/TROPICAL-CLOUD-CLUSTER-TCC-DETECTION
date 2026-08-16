import json
import sys

def main():
    notebook_path = "TCC.ipynb"
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # 1. Remove the %pip install cell
    # It is a code cell containing %pip install -q tensorflow keras gradio
    cells = nb.get("cells", [])
    new_cells = []
    for cell in cells:
        if cell.get("cell_type") == "code":
            source = "".join(cell.get("source", []))
            if "%pip install -q tensorflow keras gradio" in source:
                continue # Skip this cell (unwanted content)
            
            # 2. Change dataset_path = "dataset" to "../dataset"
            if "dataset_path = \"dataset\"" in source:
                cell["source"] = [line.replace("dataset_path = \"dataset\"", "dataset_path = \"../dataset\"") for line in cell["source"]]
                
        new_cells.append(cell)
        
    nb["cells"] = new_cells
    
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)

if __name__ == "__main__":
    main()
