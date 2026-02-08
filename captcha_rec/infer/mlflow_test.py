import requests


def test_api():
    with open("test.jpg", "rb") as f:
        addr = "http://localhost:8000/predict"
        response = requests.post(addr, files={"file": f})

    if response.status_code == 200:
        result = response.json()
        print("   Prediction successful!")
        print(f"   Filename: {result['filename']}")
        print(f"   Prediction: {result['prediction']}")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)


def health_api():
    response = requests.get(
        "http://localhost:8000/health",
    )

    if response.status_code == 200:
        print("   Healthy!")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)


if __name__ == "__main__":
    test_api()
    health_api()
