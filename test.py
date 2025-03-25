import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import string

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Parameters
num_students = 100
courses = [
    {"name": "Python Programming", "id": "CS101", "min_score": 0, "max_score": 100},
    {"name": "Data Structures", "id": "CS102", "min_score": 0, "max_score": 100},
    {"name": "Database Management", "id": "CS103", "min_score": 0, "max_score": 100},
    {"name": "Web Development", "id": "CS104", "min_score": 0, "max_score": 100},
    {"name": "Machine Learning", "id": "CS105", "min_score": 0, "max_score": 100},
    {"name": "Algorithms", "id": "CS106", "min_score": 0, "max_score": 100},
    {"name": "Software Engineering", "id": "CS107", "min_score": 0, "max_score": 100},
    {"name": "Computer Networks", "id": "CS108", "min_score": 0, "max_score": 100}
]

# Generate student data
def generate_email(name):
    # Replace spaces with dots and add random digits
    email_name = name.lower().replace(' ', '.')
    domain = random.choice(['gmail.com', 'outlook.com', 'student.edu', 'college.org'])
    return f"{email_name}{random.randint(1, 999)}@{domain}"

def generate_student_names(num):
    first_names = ['John', 'Emma', 'Liam', 'Olivia', 'Noah', 'Ava', 'Michael', 'Sophia', 
                   'Raj', 'Priya', 'Wei', 'Ming', 'Elena', 'Carlos', 'Fatima', 'Ahmed',
                   'James', 'Maria', 'David', 'Sarah', 'Jacob', 'Emily', 'William', 'Chloe',
                   'Aiden', 'Sofia', 'Ethan', 'Isabella', 'Daniel', 'Mia', 'Matthew', 'Charlotte',
                   'Joseph', 'Amelia', 'Lucas', 'Harper', 'Samuel', 'Evelyn', 'Henry', 'Abigail']
    
    last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis',
                  'Patel', 'Kim', 'Chen', 'Wang', 'Rodriguez', 'Martinez', 'Khan', 'Ali',
                  'Wilson', 'Anderson', 'Thomas', 'Taylor', 'Moore', 'Jackson', 'Martin', 'Lee',
                  'Singh', 'Kumar', 'Liu', 'Zhang', 'Gonzalez', 'Hernandez', 'Hassan', 'Ahmed',
                  'Clark', 'Lewis', 'Robinson', 'Walker', 'Young', 'Allen', 'King', 'Wright']
    
    names = []
    for _ in range(num):
        first = random.choice(first_names)
        last = random.choice(last_names)
        names.append(f"{first} {last}")
    
    return names

def assign_grade(score):
    if score >= 90:
        return "A"
    elif score >= 80:
        return "B"
    elif score >= 70:
        return "C"
    elif score >= 60:
        return "D"
    else:
        return "F"

def generate_data():
    student_names = generate_student_names(num_students)
    student_emails = [generate_email(name) for name in student_names]
    
    data = []
    
    # For each student
    for i in range(num_students):
        # Each student takes each course
        student_skill_level = np.random.normal(70, 15)  # Base skill level
        
        for course in courses:
            # Determine how many attempts for this course (1-3)
            num_attempts = random.randint(1, 3)
            
            for attempt in range(1, num_attempts + 1):
                # Score tends to improve with each attempt
                improvement = (attempt - 1) * random.uniform(3, 10)
                
                # Calculate score based on student's skill level for this course
                course_difficulty = random.uniform(0.8, 1.2)
                raw_score = student_skill_level * course_difficulty + improvement
                
                # Ensure score is within bounds
                score = max(min(raw_score, course["max_score"]), course["min_score"])
                score = round(score, 1)
                
                data.append({
                    "Course Name": course["name"],
                    "Course ID": course["id"],
                    "Attempt ID": attempt,
                    "Candidate Name": student_names[i],
                    "Candidate Email": student_emails[i],
                    "Mark": score,
                    "Grade": assign_grade(score)
                })
    
    return pd.DataFrame(data)

# Generate the dataset
df = generate_data()

# Save to both CSV and Excel for the requirement
df.to_csv("student_performance_data.csv", index=False)
df.to_excel("student_performance_data.xlsx", index=False, engine='openpyxl')

print(f"Generated data for {num_students} students across {len(courses)} courses.")
print(f"Total records: {len(df)}")
print("\nSample data:")
print(df.head())