import os
import json
import pickle
import numpy as np
import tornado.web
import tornado.ioloop

if not os.path.exists('mymodel.pkl'):
    exit('No model exists!')

filename = 'mymodel.pkl'
model = pickle.load(open(filename, 'rb'))

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
    application = tornado.web.Application([
        (r'/', MainHandler),
        (r'/predict', PredictionHandler)
    ])
    application.listen(8888)
    tornado.ioloop.IOLoop.current().start()

