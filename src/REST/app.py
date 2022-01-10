"""This file defines a python machine learning application."""
import os
import json
import pickle
import numpy as np
import tornado.web
import tornado.ioloop


class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.write('Hello, Tornado!')


class PredictionHandler(tornado.web.RequestHandler):
    def post(self):
        params = self.request.arguments
        x = np.array(params['input'])
        y = int(model.predict([x])[0])
        self.write(json.dumps({'prediction': y}))
        self.finish()


if __name__ == '__main__':
    if not os.path.exists('mymodel.pkl'):
        exit('No model exists!')
        
    filename = 'mymodel.pkl'
    model = pickle.load(open(filename, 'rb'))

    application = tornado.web.Application([
        (r'/', MainHandler),
        (r'/predict', PredictionHandler)
    ])
    
    application.listen(8080)
    tornado.ioloop.IOLoop.current().start()
