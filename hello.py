from flask import Flask, redirect, url_for, request, render_template
import inspect

app = Flask(__name__)

@app.route('/')
def home():
    return '<h1>Hello World!</h1>'

## same as above
# def home():
#     return '<h1>Hello World!</h1>'
# home = app.route('/')(home) 

@app.route('/hello/<name>')
def hello(name):
    return '<h2>Hello {}!</h2>'.format(name)

@app.route('/user/<int:userID>')
def user(userID):
    return '<h2>User {}</h2>'.format(userID)

@app.route('/name/<name>')
def name(name):
    if type(name) == str:
        return redirect(url_for('hello', name=name))
    elif type(name) == int:
        return redirect(url_for('user', userID = name))
    else:
        return '<h2>Error</h2>'

@app.route('/success/<name>')
def success(name):
   return 'welcome %s' % name

@app.route('/login',methods = ['POST', 'GET'])
def login():
   if request.method == 'POST':
      user = request.form['nm']
      return redirect(url_for('success',name = user))
   else:
      user = request.args.get('nm')
      return redirect(url_for('success',name = user))





if __name__ == '__main__':
    app.run(debug=True)