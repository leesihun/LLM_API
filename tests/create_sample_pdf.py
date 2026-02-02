"""
Create sample PDF documents for RAG testing
"""
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from pathlib import Path


def create_sample_pdf(output_path: str):
    """Create a sample PDF with test content about AI and machine learning"""
    c = canvas.Canvas(output_path, pagesize=letter)
    width, height = letter

    # Page 1: Introduction to Machine Learning
    c.setFont("Helvetica-Bold", 24)
    c.drawString(1*inch, height - 1*inch, "Introduction to Machine Learning")

    c.setFont("Helvetica", 12)
    y = height - 1.5*inch

    content_page1 = [
        "Machine learning is a subset of artificial intelligence (AI) that provides",
        "systems the ability to automatically learn and improve from experience",
        "without being explicitly programmed. Machine learning focuses on the",
        "development of computer programs that can access data and use it to",
        "learn for themselves.",
        "",
        "The process of learning begins with observations or data, such as",
        "examples, direct experience, or instruction, in order to look for",
        "patterns in data and make better decisions in the future based on",
        "the examples that we provide.",
        "",
        "Types of Machine Learning:",
        "",
        "1. Supervised Learning: The algorithm learns from labeled training",
        "   data, and makes predictions based on that data. Examples include",
        "   classification and regression problems.",
        "",
        "2. Unsupervised Learning: The algorithm learns from unlabeled data",
        "   to find hidden patterns. Examples include clustering and",
        "   dimensionality reduction.",
        "",
        "3. Reinforcement Learning: The algorithm learns by interacting with",
        "   an environment and receiving rewards or penalties for actions.",
    ]

    for line in content_page1:
        c.drawString(1*inch, y, line)
        y -= 0.25*inch

    c.showPage()

    # Page 2: Deep Learning and Neural Networks
    c.setFont("Helvetica-Bold", 24)
    c.drawString(1*inch, height - 1*inch, "Deep Learning and Neural Networks")

    c.setFont("Helvetica", 12)
    y = height - 1.5*inch

    content_page2 = [
        "Deep learning is a subset of machine learning that uses neural networks",
        "with many layers (hence 'deep') to analyze various factors of data.",
        "",
        "Neural Network Architecture:",
        "",
        "A neural network consists of interconnected nodes or neurons organized",
        "in layers. The basic architecture includes:",
        "",
        "- Input Layer: Receives the initial data",
        "- Hidden Layers: Process the information through weighted connections",
        "- Output Layer: Produces the final prediction or classification",
        "",
        "Key Concepts:",
        "",
        "1. Activation Functions: Functions like ReLU, Sigmoid, and Tanh that",
        "   introduce non-linearity into the network.",
        "",
        "2. Backpropagation: Algorithm for training neural networks by",
        "   calculating gradients and updating weights.",
        "",
        "3. Optimization: Methods like SGD, Adam, and RMSprop for minimizing",
        "   the loss function during training.",
        "",
        "4. Regularization: Techniques like dropout and L2 regularization",
        "   to prevent overfitting.",
    ]

    for line in content_page2:
        c.drawString(1*inch, y, line)
        y -= 0.25*inch

    c.showPage()

    # Page 3: Natural Language Processing
    c.setFont("Helvetica-Bold", 24)
    c.drawString(1*inch, height - 1*inch, "Natural Language Processing (NLP)")

    c.setFont("Helvetica", 12)
    y = height - 1.5*inch

    content_page3 = [
        "Natural Language Processing (NLP) is a branch of AI that helps",
        "computers understand, interpret, and manipulate human language.",
        "",
        "Key NLP Tasks:",
        "",
        "1. Text Classification: Categorizing text into predefined classes",
        "   (e.g., spam detection, sentiment analysis)",
        "",
        "2. Named Entity Recognition (NER): Identifying and classifying named",
        "   entities in text (people, organizations, locations)",
        "",
        "3. Machine Translation: Automatically translating text between",
        "   languages (e.g., Google Translate)",
        "",
        "4. Question Answering: Building systems that can answer questions",
        "   posed in natural language",
        "",
        "5. Text Generation: Creating human-like text using models like GPT",
        "",
        "Modern NLP Approaches:",
        "",
        "Transformer-based models like BERT, GPT, and T5 have revolutionized",
        "NLP by using self-attention mechanisms to process sequential data",
        "more effectively than previous recurrent neural network approaches.",
        "",
        "RAG (Retrieval-Augmented Generation) combines retrieval systems with",
        "generative models to produce more accurate and factual responses.",
    ]

    for line in content_page3:
        c.drawString(1*inch, y, line)
        y -= 0.25*inch

    c.save()
    print(f"Created sample PDF: {output_path}")


def create_company_handbook_pdf(output_path: str):
    """Create a sample company handbook PDF for testing"""
    c = canvas.Canvas(output_path, pagesize=letter)
    width, height = letter

    # Page 1: Company Overview
    c.setFont("Helvetica-Bold", 24)
    c.drawString(1*inch, height - 1*inch, "TechCorp Employee Handbook")

    c.setFont("Helvetica", 12)
    y = height - 1.5*inch

    content = [
        "Welcome to TechCorp! This handbook contains important information",
        "about our company policies, benefits, and procedures.",
        "",
        "Company Mission:",
        "To innovate and deliver cutting-edge technology solutions that",
        "improve people's lives and transform businesses worldwide.",
        "",
        "Core Values:",
        "1. Innovation - We embrace creativity and new ideas",
        "2. Integrity - We act with honesty and transparency",
        "3. Collaboration - We work together as one team",
        "4. Excellence - We strive for the highest quality",
        "",
        "Work Hours:",
        "Standard work hours are 9:00 AM to 5:00 PM, Monday through Friday.",
        "Flexible work arrangements are available upon manager approval.",
        "",
        "Remote Work Policy:",
        "Employees may work remotely up to 3 days per week with manager",
        "approval. A reliable internet connection and dedicated workspace",
        "are required for remote work.",
        "",
        "Vacation Policy:",
        "Full-time employees receive 20 days of paid vacation per year.",
        "Vacation requests should be submitted at least 2 weeks in advance.",
    ]

    for line in content:
        c.drawString(1*inch, y, line)
        y -= 0.25*inch

    c.showPage()

    # Page 2: Benefits
    c.setFont("Helvetica-Bold", 24)
    c.drawString(1*inch, height - 1*inch, "Employee Benefits")

    c.setFont("Helvetica", 12)
    y = height - 1.5*inch

    content2 = [
        "Health Insurance:",
        "TechCorp provides comprehensive health insurance coverage including",
        "medical, dental, and vision plans. Coverage begins on the first",
        "day of employment.",
        "",
        "401(k) Retirement Plan:",
        "Employees can contribute to our 401(k) plan with company matching",
        "up to 4% of salary. Vesting is immediate for employee contributions",
        "and follows a 3-year graduated schedule for company match.",
        "",
        "Professional Development:",
        "Annual budget of $2,000 per employee for training, conferences,",
        "certifications, and educational materials.",
        "",
        "Parental Leave:",
        "12 weeks paid parental leave for primary caregivers",
        "4 weeks paid parental leave for secondary caregivers",
        "",
        "Contact HR:",
        "Email: hr@techcorp.example.com",
        "Phone: 555-0100",
        "Office: Building A, Room 101",
    ]

    for line in content2:
        c.drawString(1*inch, y, line)
        y -= 0.25*inch

    c.save()
    print(f"Created company handbook PDF: {output_path}")


if __name__ == "__main__":
    # Create test data directory
    test_data_dir = Path(__file__).parent / "test_data"
    test_data_dir.mkdir(exist_ok=True)

    # Create sample PDFs
    create_sample_pdf(str(test_data_dir / "ml_introduction.pdf"))
    create_company_handbook_pdf(str(test_data_dir / "company_handbook.pdf"))

    print(f"\nTest PDFs created in: {test_data_dir}")
