#!/usr/bin/env python
# coding: utf-8

# In[64]:


import re
def register():
    data=open("Userdata2.txt",'r')
    UserName=input("Please enter your username:")
    PassWord1=input("Enter your password:")
    PassWord2=input("Enter your confirm pasaswword:")
    Secretkey=input("Scecret Three digit PIN:")
    UN=re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
    PW=re. compile(r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*#?&])[A-Za-z\d@$!#%*?&]{5,16}$') 
    c=[]
    d=[]
    f=[]
    for i in data:
        a,b,e=i.split(",")
        b=b.strip()
        e=e.strip()
        c.append(a)
        d.append(b)
        f.append(e)
        data1=dict(zip(c,d))
        data2=dict(zip(c,f))
    if PassWord1== PassWord2 and (UserName not in c):
        if(re.fullmatch(UN,UserName)) and (re.search(PW,PassWord1)):
            data=open("Userdata2.txt",'a')
            data.write(UserName+","+PassWord1+","+Secretkey+"\n")
        else:
            print("Your Username and Password not as per condition")
            register()
    else:
        print("Your User name already exists OR Password and confirm password not matched")
        register()

def login():
    db=open("Userdata2.txt",'r')
    UserName=input("Please enter your username:")
    PassWord=input("Enter your password:")
    if not len(UserName or PassWord)<1:
        c=[]
        d=[]
        f=[]
        for i in db:
            a,b,e=i.split(",")
            b=b.strip()
            e=e.strip()
            c.append(a)
            d.append(b)
            f.append(e)
            data=dict(zip(c,d))
        
        
        try:
            if data[UserName]:
                try:
                    if PassWord == data[UserName]:
                        print("Success")
                        fnprint()
                        
                    else:
                        print("Password incorrect")
                        forgotpassword(UserName)
                         
                except:
                    print("Username or Password is incorrect")
            else:
                print("User Doesn't exist")
        except:
            print("Login Error") 


# In[30]:


def fnprint():
    name=input()
    details=open("Details.txt",'a')
    details.write("Great you are in "+name+'\n')


# In[62]:


def forgotpassword(UserName):
    data=open("Userdata2.txt",'r+')
    c=[]
    d=[]
    f=[]
    for i in data:
        a,b,e=i.split(",")
        b=b.strip()
        e=e.strip()
        c.append(a)
        d.append(b)
        f.append(e)
        data1=dict(zip(c,d))
        data2=dict(zip(c,f))
 
    UserName=UserName
    Secretkey=input("Please enter yoour PIN:")
    if data2[UserName]==Secretkey:
        print("Your password is:",data1[UserName])
    else:
        print("Please Enter Correct PIN")
        forgotpassword(UserName)
             
        
        


# In[36]:


def home(choice= None):
    choice=input("Please give your option, Signup or Login:")
    if(choice=="Login"):
        login()
    elif(choice=="Signup"):
        register()
    else:
        print("Please enter valid input")
home()


# In[ ]:




