import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from groq import Groq
from io import StringIO
import altair as alt
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Set page configuration
st.set_page_config(
    page_title="Student Performance Analysis",
    page_icon="📚",
    layout="wide"
)

# ---- UTILITY FUNCTIONS ----

def load_data(uploaded_file):
    """
    ANAR1: Read performance report from multiple formats (CSV, Excel)
    """
    if uploaded_file is not None:
        # Check file extension
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            return None
        
        # Validate dataframe columns
        required_cols = ['Course Name', 'Course ID', 'Attempt ID', 
                         'Candidate Name', 'Candidate Email', 'Mark', 'Grade']
        
        if not all(col in df.columns for col in required_cols):
            st.error("The file is missing required columns. Please check the file format.")
            return None
            
        return df
    return None

def analyze_performance(df):
    """
    ANAR2: Analyze strengths and weaknesses
    """
    # Get the highest attempt for each student and course
    latest_attempts = df.sort_values(['Candidate Email', 'Course ID', 'Attempt ID'])\
                        .groupby(['Candidate Email', 'Course ID'])\
                        .last()\
                        .reset_index()
    
    # Calculate average marks for each student
    student_avg = latest_attempts.groupby('Candidate Email')['Mark'].mean().reset_index()
    student_avg.columns = ['Candidate Email', 'Average Mark']
    
    # Identify strengths and weaknesses
    strengths_weaknesses = {}
    
    for email in latest_attempts['Candidate Email'].unique():
        student_data = latest_attempts[latest_attempts['Candidate Email'] == email]
        student_name = student_data['Candidate Name'].iloc[0]
        
        # Sort courses by marks (descending)
        sorted_courses = student_data.sort_values('Mark', ascending=False)
        
        # Top 3 courses are strengths, bottom 3 are weaknesses
        top_courses = sorted_courses.head(3)
        bottom_courses = sorted_courses.tail(3)
        
        strengths = list(zip(top_courses['Course Name'], top_courses['Mark']))
        weaknesses = list(zip(bottom_courses['Course Name'], bottom_courses['Mark']))
        
        strengths_weaknesses[email] = {
            'name': student_name,
            'strengths': strengths,
            'weaknesses': weaknesses,
            'avg_mark': student_avg[student_avg['Candidate Email'] == email]['Average Mark'].iloc[0]
        }
    
    return strengths_weaknesses, latest_attempts, student_avg

def classify_students(student_avg):
    """
    Classify students into performance categories
    """
    # Calculate class average
    class_avg = student_avg['Average Mark'].mean()
    class_std = student_avg['Average Mark'].std()
    
    # Define thresholds for classification
    high_threshold = class_avg + 0.5 * class_std
    low_threshold = class_avg - 0.5 * class_std
    
    # Classify students
    student_avg['Performance Category'] = 'Medium'
    student_avg.loc[student_avg['Average Mark'] >= high_threshold, 'Performance Category'] = 'High'
    student_avg.loc[student_avg['Average Mark'] <= low_threshold, 'Performance Category'] = 'Low'
    
    return student_avg, class_avg, class_std

def recommend_courses(student_data, weaknesses):
    """
    ANAR3: Recommend courses for improvement
    """
    # Create a prompt for Groq API
    prompt = f"""
    Based on the following student data and weaknesses, recommend specific courses that would help the student improve.
    
    Student Name: {student_data['name']}
    Average Mark: {student_data['avg_mark']:.2f}
    
    Weaknesses:
    {weaknesses}
    
    Please recommend 3-5 specific courses or resources that would address these weaknesses.
    Format the output as a JSON array of objects with 'course_name', 'description', and 'estimated_duration' fields.
    """
    
    # Call Groq API
    try:
        response = client.chat.completions.create(
            model="llama3-8b-8192",  # Use llama3-8b or llama3-70b based on needs
            messages=[
                {"role": "system", "content": "You are an educational advisor specialized in recommending courses to help students improve their academic performance."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=2000
        )
        
        # Extract and parse the response
        import json
        import re
        
        completion_text = response.choices[0].message.content
        
        # Try to extract JSON from the response
        json_match = re.search(r'\[\s*{.*}\s*\]', completion_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            try:
                recommendations = json.loads(json_str)
                return recommendations
            except json.JSONDecodeError:
                pass
        
        # Fallback recommendations if parsing fails
        fallback_recommendations = [
            {
                "course_name": f"Advanced {weakness[0]} Fundamentals",
                "description": f"A comprehensive course designed to strengthen your understanding of {weakness[0]}.",
                "estimated_duration": "4 weeks"
            } for weakness in student_data['weaknesses'][:3]
        ]
        
        return fallback_recommendations
        
    except Exception as e:
        st.error(f"Error calling Groq API: {e}")
        # Return fallback recommendations
        return [
            {
                "course_name": f"Advanced {weakness[0]} Fundamentals",
                "description": f"A comprehensive course designed to strengthen your understanding of {weakness[0]}.",
                "estimated_duration": "4 weeks"
            } for weakness in student_data['weaknesses'][:3]
        ]

def pair_students(student_classifications, strengths_weaknesses):
    """
    ANAR4: Pair weak performers with strong performers
    """
    # Get low and high performers
    low_performers = student_classifications[student_classifications['Performance Category'] == 'Low']
    high_performers = student_classifications[student_classifications['Performance Category'] == 'High']
    
    # Create pairs
    pairs = []
    
    for _, low_performer in low_performers.iterrows():
        low_email = low_performer['Candidate Email']
        low_weaknesses = [w[0] for w in strengths_weaknesses[low_email]['weaknesses']]
        
        for _, high_performer in high_performers.iterrows():
            high_email = high_performer['Candidate Email']
            high_strengths = [s[0] for s in strengths_weaknesses[high_email]['strengths']]
            
            # Find matching strengths and weaknesses
            matches = set(low_weaknesses).intersection(set(high_strengths))
            
            if matches:
                pairs.append({
                    'low_performer': low_email,
                    'low_performer_name': strengths_weaknesses[low_email]['name'],
                    'high_performer': high_email,
                    'high_performer_name': strengths_weaknesses[high_email]['name'],
                    'matching_subjects': list(matches)
                })
    
    return pairs

def track_improvement(df):
    """
    ANAR5: Track improvement based on attempts
    """
    # Get students who have taken multiple attempts
    multi_attempts = df[df.duplicated(['Candidate Email', 'Course ID'], keep=False)]
    
    if multi_attempts.empty:
        return pd.DataFrame(), {}
    
    # Group by student and course
    improvement_data = multi_attempts.sort_values(['Candidate Email', 'Course ID', 'Attempt ID'])
    
    # Calculate improvement
    improvement_summary = []
    
    for (email, course_id), group in improvement_data.groupby(['Candidate Email', 'Course ID']):
        if len(group) > 1:
            first_attempt = group.iloc[0]
            last_attempt = group.iloc[-1]
            
            improvement_summary.append({
                'Candidate Email': email,
                'Candidate Name': first_attempt['Candidate Name'],
                'Course Name': first_attempt['Course Name'],
                'First Attempt Mark': first_attempt['Mark'],
                'Last Attempt Mark': last_attempt['Mark'],
                'Improvement': last_attempt['Mark'] - first_attempt['Mark'],
                'Attempts': len(group)
            })
    
    improvement_df = pd.DataFrame(improvement_summary)
    
    # Calculate overall improvement by student
    if not improvement_df.empty:
        student_improvement = improvement_df.groupby(['Candidate Email', 'Candidate Name'])['Improvement'].mean().reset_index()
        student_improvement.columns = ['Candidate Email', 'Candidate Name', 'Average Improvement']
        return improvement_df, student_improvement
    
    return pd.DataFrame(), {}

# ---- MAIN APP ----

def main():
    st.title("Student Performance Analysis & Recommendation Tool")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Upload Data", "Individual Analysis", "Class Performance", "Pair Recommendations", "Improvement Tracking"])
    
    # Initialize session state
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'strengths_weaknesses' not in st.session_state:
        st.session_state.strengths_weaknesses = None
    if 'latest_attempts' not in st.session_state:
        st.session_state.latest_attempts = None
    if 'student_avg' not in st.session_state:
        st.session_state.student_avg = None
    if 'student_classifications' not in st.session_state:
        st.session_state.student_classifications = None
    if 'pairs' not in st.session_state:
        st.session_state.pairs = None
    if 'improvement_data' not in st.session_state:
        st.session_state.improvement_data = None
    if 'student_improvement' not in st.session_state:
        st.session_state.student_improvement = None
    
    # Upload Data Page
    if page == "Upload Data":
        st.header("Upload Your Data")
        
        uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=['csv', 'xlsx', 'xls'])
        
        if uploaded_file is not None:
            st.session_state.data = load_data(uploaded_file)
            
            if st.session_state.data is not None:
                st.success("Data loaded successfully!")
                
                # Display sample data
                st.subheader("Sample Data")
                st.dataframe(st.session_state.data.head())
                
                # Process data
                st.session_state.strengths_weaknesses, st.session_state.latest_attempts, st.session_state.student_avg = analyze_performance(st.session_state.data)
                st.session_state.student_classifications, class_avg, class_std = classify_students(st.session_state.student_avg)
                st.session_state.pairs = pair_students(st.session_state.student_classifications, st.session_state.strengths_weaknesses)
                st.session_state.improvement_data, st.session_state.student_improvement = track_improvement(st.session_state.data)
                
                # Display summary statistics
                st.subheader("Summary Statistics")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Students", len(st.session_state.student_avg))
                
                with col2:
                    st.metric("Class Average", f"{class_avg:.2f}")
                
                with col3:
                    st.metric("Standard Deviation", f"{class_std:.2f}")
    
    # Individual Analysis Page
    elif page == "Individual Analysis":
        st.header("Individual Student Analysis")
        
        if st.session_state.data is None:
            st.warning("Please upload data first!")
            return
        
        # Select student
        student_emails = sorted(st.session_state.strengths_weaknesses.keys())
        selected_email = st.selectbox("Select Student", student_emails)
        
        if selected_email:
            student_data = st.session_state.strengths_weaknesses[selected_email]
            
            st.subheader(f"Analysis for {student_data['name']}")
            
            # Create columns for strengths and weaknesses
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Strengths")
                for course, mark in student_data['strengths']:
                    st.write(f"- {course}: {mark:.2f}")
            
            with col2:
                st.subheader("Weaknesses")
                weaknesses_text = ""
                for course, mark in student_data['weaknesses']:
                    weaknesses_text += f"- {course}: {mark:.2f}\n"
                    st.write(f"- {course}: {mark:.2f}")
            
            # Get course recommendations
            st.subheader("Recommended Courses")
            with st.spinner("Generating recommendations..."):
                recommendations = recommend_courses(student_data, weaknesses_text)
            
            for i, rec in enumerate(recommendations):
                with st.expander(f"{i+1}. {rec['course_name']}"):
                    st.write(f"**Description:** {rec['description']}")
                    st.write(f"**Estimated Duration:** {rec['estimated_duration']}")
            
            # Show performance over time if multiple attempts exist
            student_attempts = st.session_state.data[st.session_state.data['Candidate Email'] == selected_email]
            
            if len(student_attempts) > len(student_attempts.drop_duplicates(['Course ID'])):
                st.subheader("Performance Improvement")
                
                # Group by course and attempt
                courses = student_attempts['Course ID'].unique()
                
                for course in courses:
                    course_attempts = student_attempts[student_attempts['Course ID'] == course].sort_values('Attempt ID')
                    course_name = course_attempts['Course Name'].iloc[0]
                    
                    if len(course_attempts) > 1:
                        # Create chart
                        chart_data = pd.DataFrame({
                            'Attempt': course_attempts['Attempt ID'],
                            'Mark': course_attempts['Mark']
                        })
                        
                        chart = alt.Chart(chart_data).mark_line(point=True).encode(
                            x='Attempt:O',
                            y=alt.Y('Mark:Q', scale=alt.Scale(domain=[0, 100])),
                            tooltip=['Attempt:O', 'Mark:Q']
                        ).properties(
                            title=f"{course_name} ({course}) Improvement",
                            width=600,
                            height=300
                        )
                        
                        st.altair_chart(chart)
    
    # Class Performance Page
    elif page == "Class Performance":
        st.header("Class Performance Dashboard")
        
        if st.session_state.data is None:
            st.warning("Please upload data first!")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Performance distribution
            st.subheader("Performance Distribution")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(st.session_state.student_avg['Average Mark'], kde=True, ax=ax)
            ax.set_xlabel('Average Mark')
            ax.set_ylabel('Number of Students')
            ax.set_title('Distribution of Student Performance')
            
            # Add vertical lines for classification thresholds
            class_avg = st.session_state.student_avg['Average Mark'].mean()
            class_std = st.session_state.student_avg['Average Mark'].std()
            
            ax.axvline(class_avg - 0.5 * class_std, color='r', linestyle='--', alpha=0.7, label='Low Threshold')
            ax.axvline(class_avg + 0.5 * class_std, color='g', linestyle='--', alpha=0.7, label='High Threshold')
            ax.legend()
            
            st.pyplot(fig)
        
        with col2:
            # Performance categories
            st.subheader("Performance Categories")
            
            category_counts = st.session_state.student_classifications['Performance Category'].value_counts()
            
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff','#99ff99'])
            ax.axis('equal')
            ax.set_title('Student Performance Categories')
            
            st.pyplot(fig)
        
        # Course performance
        st.subheader("Course Performance")
        
        course_performance = st.session_state.latest_attempts.groupby('Course Name')['Mark'].agg(['mean', 'min', 'max', 'count']).reset_index()
        course_performance.columns = ['Course Name', 'Average Mark', 'Minimum Mark', 'Maximum Mark', 'Number of Students']
        
        # Sort by average mark
        course_performance = course_performance.sort_values('Average Mark', ascending=False)
        
        # Create a bar chart
        chart = alt.Chart(course_performance).mark_bar().encode(
            x=alt.X('Course Name:N', sort=None),
            y='Average Mark:Q',
            color=alt.Color('Average Mark:Q', scale=alt.Scale(scheme='blues')),
            tooltip=['Course Name', 'Average Mark', 'Minimum Mark', 'Maximum Mark', 'Number of Students']
        ).properties(
            width=800,
            height=400
        )
        
        st.altair_chart(chart)
        
        # Show course performance table
        st.subheader("Course Performance Details")
        st.dataframe(course_performance.style.highlight_max(axis=0, color='lightgreen').highlight_min(axis=0, color='#ff9999'))
    
    # Pair Recommendations Page
    elif page == "Pair Recommendations":
        st.header("Student Pairing Recommendations")
        
        if st.session_state.data is None:
            st.warning("Please upload data first!")
            return
        
        if not st.session_state.pairs:
            st.info("No suitable pairs found. This could happen if there are no clear high and low performers or if their strengths and weaknesses don't align.")
            return
        
        # Display pairs
        for i, pair in enumerate(st.session_state.pairs):
            with st.expander(f"Pair {i+1}: {pair['low_performer_name']} + {pair['high_performer_name']}"):
                st.write(f"**Low Performer:** {pair['low_performer_name']} ({pair['low_performer']})")
                st.write(f"**High Performer:** {pair['high_performer_name']} ({pair['high_performer']})")
                st.write("**Matching Subjects:**")
                for subject in pair['matching_subjects']:
                    st.write(f"- {subject}")
    
    # Improvement Tracking Page
    elif page == "Improvement Tracking":
        st.header("Student Improvement Tracking")
        
        if st.session_state.data is None:
            st.warning("Please upload data first!")
            return
        
        if st.session_state.improvement_data.empty:
            st.info("No improvement data available. This could happen if students haven't taken multiple attempts for any courses.")
            return
        
        # Top improvers
        st.subheader("Top Improvers")
        
        top_improvers = st.session_state.student_improvement.sort_values('Average Improvement', ascending=False).head(10)
        
        chart = alt.Chart(top_improvers).mark_bar().encode(
            x=alt.X('Average Improvement:Q'),
            y=alt.Y('Candidate Name:N', sort='-x'),
            color=alt.Color('Average Improvement:Q', scale=alt.Scale(scheme='greens')),
            tooltip=['Candidate Name', 'Average Improvement']
        ).properties(
            width=700,
            height=400,
            title='Top 10 Most Improved Students'
        )
        
        st.altair_chart(chart)
        
        # Course-specific improvements
        st.subheader("Course-Specific Improvements")
        
        course_improvements = st.session_state.improvement_data.groupby('Course Name')['Improvement'].mean().reset_index()
        course_improvements = course_improvements.sort_values('Improvement', ascending=False)
        
        chart = alt.Chart(course_improvements).mark_bar().encode(
            x=alt.X('Course Name:N', sort='-y'),
            y=alt.Y('Improvement:Q'),
            color=alt.Color('Improvement:Q', scale=alt.Scale(scheme='blues')),
            tooltip=['Course Name', 'Improvement']
        ).properties(
            width=700,
            height=400,
            title='Average Improvement by Course'
        )
        
        st.altair_chart(chart)
        
        # Detailed improvement data
        st.subheader("Detailed Improvement Data")
        st.dataframe(st.session_state.improvement_data.sort_values('Improvement', ascending=False))

if __name__ == "__main__":
    main()

    #changes