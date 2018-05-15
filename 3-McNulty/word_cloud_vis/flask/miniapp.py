from flask import Flask, render_template, flash, request
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField

from flask_fn import test_new_comment
 
# App config.
DEBUG = True
app = Flask(__name__,static_url_path='')
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'
 
class ReusableForm(Form):
    username = TextField('Userame:', validators=[validators.required()])
    flair = TextField('Flair')
    comment = TextField('Comment:', validators=[validators.required()])
 
    def reset(self):
        blankData = MultiDict([ ('csrf', self.reset_csrf() ) ])
        self.process(blankData)

@app.route("/")
def render_index():
    return render_template("index.html")

@app.route("/upv")
def render_upv():
    return render_template('upv.html') 

@app.route("/upv/<string:page_name>")
def render_upv_sub(page_name):
    return render_template('upv/%s' % page_name)

@app.route("/dnv")
def render_dnv():
    return render_template('dnv.html') 

@app.route("/dnv/<string:page_name>")
def render_dnv_sub(page_name):
    return render_template('dnv/%s' % page_name)

@app.route("/dmu")
def render_dmu():
    return render_template('dmu.html') 

@app.route("/dmu/<string:page_name>")
def render_dmu_sub(page_name):
    return render_template('dmu/%s' % page_name)

@app.route("/umd")
def render_umd():
    return render_template('umd.html') 

@app.route("/umd/<string:page_name>")
def render_umd_sub(page_name):
    return render_template('umd/%s' % page_name)
    
@app.route("/flask", methods=['GET', 'POST'])
def hello():
    form = ReusableForm(request.form)
 
    print(form.errors)
    if request.method == 'POST':
        username=request.form['username']
        flair=request.form['flair']
        comment=request.form['comment']
        
 
        if form.validate():
            # Save the comment here.
            flash(test_new_comment(username,flair,comment))
        else:
            flash('Error: Username and comment fields are required. ')
 
    return render_template('flask.html', form=form)



@app.route("/<string:page_name>")
def render_static(page_name):
     return render_template('%s' % page_name)



if __name__ == "__main__":
    app.run(host='0.0.0.0',port=80)