from flask import Flask, render_template, flash, redirect, url_for, session, request, logging
from flask import Flask
from flask_mysqldb import MySQL
from wtforms import Form, StringField, TextAreaField, PasswordField, validators
from passlib.hash import sha256_crypt
from functools import wraps
from flask_login import login_user,LoginManager
import twpre_cls as ml
import random


app =  Flask(__name__)
#generates a random string
def creuid():
    num1=0
    num2=0
    num1= random.randint(9999,99999)
    num2= random.randint(9999,99999)
    digits = len(str(num2))
    num1 = num1 * (10**digits)
    num1 += num2
    print(num1)
    return num1

num= random.randint(0, 1000)


app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'your username here' #MySQL username
app.config['MYSQL_PASSWORD'] = 'your password here' #MySQl password
app.config['MYSQL_DB'] = 'cybergaruna' #MySql database 
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'
mysql = MySQL(app)



#adding Login Manager/login required class
def login_required(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if 'logged_in' in session:
            return f(*args, **kwargs)
        else:
            flash('Unauthorized, Please login')
            return redirect(url_for('login'))

    return wrap
#home
@app.route('/',methods=['GET', 'POST'])
def home():
    #get the count of predicted tweets
    count=ml.predt()
    return render_template('index.html', count=count)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Get Form Fields
        username = request.form['username']
        password_candidate = request.form['password']

        # Create cursor
        cur = mysql.connection.cursor()

        # Get user by username
        result = cur.execute("SELECT * FROM users WHERE username = %s", [username])

        if result > 0:
            # Get stored hash
            data = cur.fetchone()
            password = data['password']

            # Compare Passwords

            #if password_candidate==password :
            if sha256_crypt.verify(password_candidate, password):
                # Passed
                session['logged_in'] = True
                session['username'] = username
                return redirect(url_for('dashboard'))
            else:
                flash("Check the Username and Password")
                return render_template('login.html')
            # Close connection
            cur.close()
        else:
            error = 'Username not found'
            return render_template('login.html', error=error)

    return render_template('login.html')



@app.route('/logout', methods=['GET', 'POST'])
@login_required
def logout():
    #logout
    if request.method == 'POST':
        session.clear()
        flash("Successfully Logged Out")
        return redirect(url_for('login'))
 
 
 #dashboard
@app.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    if request.method == 'POST':
        cur = mysql.connection.cursor()
        result = cur.execute("SELECT * FROM regcases")   
        if result > 0:   
            datam = cur.fetchall()
        return render_template('sampletable.html',data1=datam)
    else:
        cur = mysql.connection.cursor()
        result = cur.execute("SELECT * FROM regcases")   
        if result > 0:   
            datam = cur.fetchall()
        return render_template('sampletable.html',data1=datam)
        
@app.route('/accepted', methods=['GET', 'POST'])
@login_required
#table of accepted cases
def accepted():
    if request.method == 'POST':
        cur = mysql.connection.cursor()
        result = cur.execute("SELECT * FROM acceptedcomplaints")   
        if result > 0:   
            data = cur.fetchall()
        return render_template('accepted_cases.html',data1 =data)

#table of rejected cases
@app.route('/rejected', methods=['GET', 'POST'])
@login_required
def rejected():
    if request.method == 'POST':
        cur = mysql.connection.cursor()
        result = cur.execute("SELECT * FROM rejectedcomplaints")   
        if result > 0:   
            data = cur.fetchall()
        return render_template('rejected_cases.html',data1 =data)


#table of registered cases
@app.route('/register_complaint')
def regcomp():
    return render_template('regcomp.html')

#submit complaint
@app.route('/submit_complaint', methods=['GET', 'POST'])
def sub_comp():
    if request.method == 'POST':
        unid=creuid()
        case_type=request.form['ctype']
        case_desc=request.form['cdesc']
        case_proof=request.form['clink']
        cur = mysql.connection.cursor()
        cur.execute("INSERT INTO regcases(id, ctype, cdescription, clink) VALUES(%s, %s, %s, %s)", (unid, case_type, case_desc, case_proof))
        mysql.connection.commit()
        cur.close()
        return render_template('reg_success.html',unid=str(unid))



@app.route('/check_complaint')

def chkcomp():
    return render_template('check.html')


#about
@app.route('/about')

def about():
    return render_template('about.html')

#view complaints
@app.route('/view', methods=['GET', 'POST'])
@login_required
def getid():
    if request.method == 'POST':
        getval=request.form['getval']
        cur = mysql.connection.cursor()
        result = cur.execute("SELECT * FROM regcases WHERE id = %s", [getval])
        if result > 0:
            data = cur.fetchone()
            case_id=data['id']
            case_type=data['ctype']
            case_desc=data['cdescription']
            case_proof=data['clink']    
    return render_template('viewcase.html',getval =case_id, case_type=case_type,case_desc=case_desc,case_proof=case_proof)
 
 
 #add to accepted table
    
@app.route('/comp_add_res', methods=['GET', 'POST'])
@login_required
def addcse():
    if request.method == 'POST':
        getval=request.form['compres']
        cur = mysql.connection.cursor()
        result = cur.execute("SELECT * FROM regcases WHERE id = %s", [getval])
        if result > 0:
            data = cur.fetchone()
            case_id=data['id']
            case_type=data['ctype']
            case_desc=data['cdescription']
            case_proof=data['clink'] 
            officer_name=session['username']
            cur.execute("DELETE FROM regcases WHERE id = %s", [getval])
            cur.execute("INSERT INTO acceptedcomplaints(id, ctype, cdescription, clink, officer) VALUES(%s, %s, %s, %s, %s)", (case_id, case_type, case_desc, case_proof, officer_name))
            mysql.connection.commit()
            cur.close()
    cur = mysql.connection.cursor()
    flash('Case is accepted!', 'success')
    result = cur.execute("SELECT * FROM regcases")    
    if result > 0:   
        data = cur.fetchall()
        return render_template('sampletable.html',data1 =data)
    else:
        flash("No cases found", 'danger')
        return render_template('sampletable.html')


#add to rejected table
@app.route('/comp_del_res', methods=['GET', 'POST'])
@login_required
def dellcse():
    
    if request.method == 'POST':
        getval=request.form['compres']
        cur = mysql.connection.cursor()
        result = cur.execute("SELECT * FROM regcases WHERE id = %s", [getval])
        if result > 0:
            data = cur.fetchone()
            case_id=data['id']
            case_type=data['ctype']
            case_desc=data['cdescription']
            case_proof=data['clink']
            officer_name=session['username'] 
            cur.execute("DELETE FROM regcases WHERE id = %s", [getval])
            cur.execute("INSERT INTO rejectedcomplaints(id, ctype, cdescription, clink, officer) VALUES(%s, %s, %s, %s, %s)", (case_id, case_type, case_desc, case_proof, officer_name))
            flash('Case is rejected!', 'success')
            mysql.connection.commit()
            cur.close()
        else:
            flash("Case not found", 'danger')
            return render_template('sampletable.html')
    cur = mysql.connection.cursor()
    result = cur.execute("SELECT * FROM regcases")   
    if result > 0:   
        data = cur.fetchall()
        return render_template('sampletable.html',data1 =data)
    else:
        flash("No cases found", 'danger')
        return render_template('sampletable.html')

          
#check status            
@app.route('/chkstatus', methods=['GET', 'POST'])
def chkstatus():
        if request.method == 'POST':
            getid=request.form['getid']
            cur = mysql.connection.cursor()
            result = cur.execute("SELECT * FROM regcases WHERE id = %s", [getid])
            if result > 0:
                flash('Your case is yet to be checked', 'success')
                return render_template('check.html')
            
            result = cur.execute("SELECT * FROM acceptedcomplaints WHERE id = %s", [getid])
            if result > 0:
                flash('Your case is under review', 'success')
                return render_template('check.html')
                
            result = cur.execute("SELECT * FROM rejectedcomplaints WHERE id = %s", [getid])
            if result > 0:
                flash('Your case is rejected', 'danger')
                return render_template('check.html')
        
    
if __name__=="__main__":
    
    app.secret_key='secret123'
    app.run(host='127.0.0.1', debug=True)
