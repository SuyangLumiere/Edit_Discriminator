import json

class ResultLogger:
    def __init__(self, save_path="results.jsonl"):
        self.save_path = save_path

    def __call__(self, data):
        self.batch_log([data])

    def batch_log(self, results):
        with open(self.save_path, "a", encoding="utf-8") as f:
            for item in results:
                f.write(ResultLogger.to_json(item)+ "\n")
    
    @staticmethod
    def to_json(result) -> str:
        return json.dumps(result.to_dict(), ensure_ascii=False) if hasattr(result, "to_dict") else json.dumps(result, ensure_ascii=False)