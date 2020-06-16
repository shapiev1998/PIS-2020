predicted = np.round(model.predict(x_test_seq))
print(classification_report(y_test, predicted, digits=5))