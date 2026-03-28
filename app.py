import streamlit as st 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# our database
faq_data = [
    #General Information
    {"cat": "General", "q": "What is SUK Technologies and what do you do?", "a": "SUK Technologies is a premier IT solutions provider specializing in AI development, Cloud computing, and Custom Software architecture."},
    {"cat": "General", "q": "Where is SUK Technologies located?", "a": "Our headquarters are located in the Tech District, with satellite offices handling global operations remotely."},
    
    #Account & Login
    {"cat": "Account", "q": "I am having trouble logging in or forgot my password.", "a": "You can reset your password via the 'Forgot Password' link on the login page. For account lockouts, contact admin@suktech.com."},
    {"cat": "Account", "q": "How do I create a new client account?", "a": "New accounts are created by our onboarding team once a service contract is signed."},
    
    #Products & Services
    {"cat": "Products", "q": "What products or services do you offer?", "a": "We offer Full-Stack Web Development, Mobile App Development (iOS/Android), and Enterprise AI Integration."},
    
    #Payment & Pricing
    {"cat": "Payment", "q": "What are your payment methods and pricing models?", "a": "We accept Wire Transfers, PayPal, and Major Credit Cards. We offer Fixed-Price project billing and Hourly-Rate consultations."},
    
    #Technical Support
    {"cat": "Support", "q": "How do I get technical support for my software?", "a": "Please raise a ticket through our Support Portal or email our 24/7 helpdesk at tech-support@suktech.com."},
    
    #Project Status
    {"cat": "Projects", "q": "How can I check my project status or progress?", "a": "Clients can view real-time milestones and sprint progress through the SUK Project Dashboard."},
    
    #Policies
    {"cat": "Policies", "q": "What is your privacy and refund policy?", "a": "We are GDPR compliant. Our refund policy depends on the milestone stage completed; please refer to your Service Level Agreement (SLA)."},
    
    #Careers & Internships
    {"cat": "Careers", "q": "Are you hiring? How do I apply for a job or internship?", "a": "We are always looking for talent! Visit suktech.com/careers to see open positions for Developers, Designers, and Interns."},
    
    #Training & Learning
    {"cat": "Training", "q": "Do you offer training, learning programs, or certifications?", "a": "Yes! SUK Academy offers internal and external training in Python, React, and Cloud Architecture certifications."}
]

# create a simple list of questions
faq_questions = [item["q"] for item in faq_data]


def find_best_answer(user_input):
    # convert text into numbers
    # Tfidf removes words like "is" , "the", "a"
    vectorizer = TfidfVectorizer(stop_words = "english")
    
    #combine questions and user_input
    all_text = faq_questions + [user_input]
    tfidf_matrix = vectorizer.fit_transform(all_text)

    #compare the user_input to the faq_data
    scores = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])

    #get index of the best score
    best_index = scores.argmax()
    confidence = scores[0][best_index]

    #only return answer if it's a good match(above 0.2 score)
    if confidence > 0.2:
        category = faq_data[best_index]["cat"]
        answer = faq_data[best_index]["a"]
        return f"[{category}] {answer}"
    else:
        return "I'm sorry, I don't have information on that. Please contact support@suktech.com."


st.title("SUK Technologies Help Desk")
# create a text box for user
user_query = st.text_input("Ask me a question about SUK")

if user_query:
    reply = find_best_answer(user_query)

    # show answer
    st.write(f"Bot: {reply}")