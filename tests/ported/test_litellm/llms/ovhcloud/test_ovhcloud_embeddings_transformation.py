from unittest.mock import patch

import litelm

model="ovhcloud/BGE-M3"

def mock_embedding_response(*args, **kwargs):
    class MockResponse:
        def __init__(self):
            self.data = [{"embedding": [0.1, 0.2, 0.3]}]
            self.usage = litelm.Usage()
            self.model = kwargs.get("model", model)
            self.object = "embedding"

        def __getitem__(self, key):
            return getattr(self, key)

    return MockResponse()


def test_ovhcloud_embeddings():
    with patch("litelm.embedding", side_effect=mock_embedding_response) as mock_embed:
        response = litelm.embedding(
            model,
            input=["good morning from litelm"],
        )

        mock_embed.assert_called_once_with(
            model,
            input=["good morning from litelm"],
        )

        assert isinstance(response.data, list)
        assert "embedding" in response.data[0]
        assert isinstance(response.data[0]["embedding"], list)
        assert response.model == model
        assert response.object == "embedding"
