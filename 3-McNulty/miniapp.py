from flask import Flask, render_template, flash, request
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField

from flask_fn import test_new_comment
 
# App config.
DEBUG = True
app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'
 
class ReusableForm(Form):
    username = TextField('Userame:', validators=[validators.required()])
    flair = TextField('Flair')
    comment = TextField('Comment:', validators=[validators.required()])
 
    def reset(self):
        blankData = MultiDict([ ('csrf', self.reset_csrf() ) ])
        self.process(blankData)
 
@app.route("/", methods=['GET', 'POST'])


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
 
    return render_template('hello.html', form=form)
 
if __name__ == "__main__":
    app.run()