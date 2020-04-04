from ceserver.server import Protocol, CEServer, CEModel
from tornado.testing import AsyncHTTPTestCase
from typing import List, Dict
import json
import requests_mock


class TestProtocol(AsyncHTTPTestCase):
    def get_app(self):
        server = CEServer(Protocol.seldon_http)
        return server.create_application()

    def test_seldon_protocol(self):
        response = self.fetch("/protocol")
        self.assertEqual(response.code, 200)
        self.assertEqual(response.body.decode("utf-8"), str(Protocol.seldon_http.value))


customHeaderKey = "Seldonheader"
customHeaderVal = "SeldonValue"


class DummyModel(CEModel):
    @staticmethod
    def getResponse() -> Dict:
        return {"foo": 1}

    def load(self):
        pass

    def process_event(self, inputs: List, headers: Dict) -> Dict:
        assert headers[customHeaderKey] == customHeaderVal
        return DummyModel.getResponse()


class TestModel(AsyncHTTPTestCase):
    def setupEnv(self):
        self.replyUrl = "http://reply-location"
        self.eventSource = "x.y.z"
        self.eventType = "a.b.c"

    def get_app(self):
        self.setupEnv()
        server = CEServer(
            Protocol.seldon_http, 9000, self.replyUrl, self.eventType, self.eventSource
        )
        model = DummyModel("name")
        server.register_model(model)
        return server.create_application()

    def test_seldon_protocol(self):
        data = {"data": {"ndarray": [[1, 2, 3]]}}
        dataStr = json.dumps(data)
        with requests_mock.Mocker() as m:
            m.post(self.replyUrl, text="resp")

            response = self.fetch(
                "/",
                method="POST",
                body=dataStr,
                headers={customHeaderKey: customHeaderVal},
            )
            self.assertEqual(response.code, 200)
            expectedResponse = json.dumps(DummyModel.getResponse())
            self.assertEqual(response.body.decode("utf-8"), expectedResponse)
            self.assertEqual(
                m.request_history[0].json(), json.dumps(DummyModel.getResponse())
            )
            headers: Dict = m.request_history[0]._request.headers
            self.assertEqual(headers["ce-source"], self.eventSource)
            self.assertEqual(headers["ce-type"], self.eventType)
