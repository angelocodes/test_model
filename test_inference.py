from inference import predict_sentiment 

def test_predict_positive():
    result = predict_sentiment("I absolutely love this movie! It was fantastic.")
    assert result == "positive"

def test_predict_negative():
    result = predict_sentiment("This is the worst movie I have ever watched.")
    assert result == "negative"


