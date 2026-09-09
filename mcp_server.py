import sys
import json
from client import SparseSpMMCSR

def handle_call(name, arguments):
    if name == "spmm":
        dense_a = arguments["dense_a"]
        dense_x = arguments["dense_x"]
        sp = SparseSpMMCSR.from_dense(dense_a)
        y = sp.multiply_dense(dense_x)
        return {"result": y, "nnz": len(sp.values)}
    return {"error": f"Unknown tool: {name}"}

def main():
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            req = json.loads(line)
            res = handle_call(req.get("name"), req.get("arguments", {}))
            print(json.dumps({"id": req.get("id"), "result": res}))
            sys.stdout.flush()
        except Exception as e:
            print(json.dumps({"error": str(e)}))
            sys.stdout.flush()

if __name__ == "__main__":
    main()
