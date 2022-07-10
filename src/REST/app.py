"""This file defines a python machine learning application."""
import os
import json
import pickle
import pathlib
import numpy as np
import tornado.web
import tornado.ioloop


class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.write("Hello, Tornado!")


class PredictionHandler(tornado.web.RequestHandler):
    def initialize(self, model):
        self.model = model

    def post(self):
        params = self.request.arguments
        x = np.array(params["input"])
        y = int(self.model.predict([x])[0])
        self.write(json.dumps({"prediction": y}))
        self.finish()


if __name__ == "__main__":
    path = pathlib.Path(__file__).parent.absolute()
    file = f"{path}/mymodel.pkl"
    if not os.path.exists(file):
        exit("No model exists!")

    model = pickle.load(open(file, "rb"))
    application = tornado.web.Application(
        [(r"/", MainHandler), (r"/predict", PredictionHandler, {"model": model})]
    )

    application.listen(8080)
    tornado.ioloop.IOLoop.current().start()
