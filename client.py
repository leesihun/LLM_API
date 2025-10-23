import httpx


class LLMApiClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self.base_url = base_url.rstrip("/")
        self.token = None

    def _headers(self):
        h = {"Content-Type": "application/json"}
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        return h

    def signup(self, username: str, password: str, role: str = "guest"):
        r = httpx.post(f"{self.base_url}/api/auth/signup", json={
            "username": username, "password": password, "role": role
        })
        r.raise_for_status()
        return r.json()

    def login(self, username: str, password: str):
        r = httpx.post(f"{self.base_url}/api/auth/login", json={
            "username": username, "password": password
        })
        r.raise_for_status()
        data = r.json()
        self.token = data["access_token"]
        return data

    def list_models(self):
        r = httpx.get(f"{self.base_url}/v1/models", headers=self._headers())
        r.raise_for_status()
        return r.json()

    def change_model(self, model: str):
        r = httpx.post(f"{self.base_url}/api/admin/model", json={"model": model}, headers=self._headers())
        r.raise_for_status()
        return r.json()

    def chat_new(self, model: str, user_message: str, agent_type: str = "auto"):
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": user_message}],
            "agent_type": agent_type
        }
        r = httpx.post(f"{self.base_url}/v1/chat/completions", json=payload, headers=self._headers())
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"], data["x_session_id"]

    def chat_continue(self, model: str, session_id: str, user_message: str, agent_type: str = "auto"):
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": user_message}],
            "session_id": session_id,
            "agent_type": agent_type
        }
        r = httpx.post(f"{self.base_url}/v1/chat/completions", json=payload, headers=self._headers())
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"], data["x_session_id"]

    def chat_sessions(self):
        r = httpx.get(f"{self.base_url}/api/chat/sessions", headers=self._headers())
        r.raise_for_status()
        return r.json()["sessions"]

    def chat_history(self, session_id: str):
        r = httpx.get(f"{self.base_url}/api/chat/history/{session_id}", headers=self._headers())
        r.raise_for_status()
        return r.json()["messages"]

    def tools(self):
        r = httpx.get(f"{self.base_url}/api/tools/list", headers=self._headers())
        r.raise_for_status()
        return r.json()["tools"]

    def math(self, expression: str):
        r = httpx.post(f"{self.base_url}/api/tools/math", json={"expression": expression}, headers=self._headers())
        r.raise_for_status()
        return r.json()["result"]

    def websearch(self, query: str, max_results: int = 5):
        r = httpx.post(f"{self.base_url}/api/tools/websearch", json={"query": query, "max_results": max_results}, headers=self._headers())
        r.raise_for_status()
        return r.json()["results"]

    def answer_from_json(self, model: str, json_blob: dict, question: str):
        prompt = f"Given this JSON: {json_blob}\nAnswer: {question}"
        return self.chat_new(model, prompt)[0]


if __name__ == "__main__":
    client = LLMApiClient()
    print("Client ready. See README.md for examples.")


