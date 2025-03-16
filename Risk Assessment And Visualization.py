import matplotlib.pyplot as plt


def simple_risk_analysis(document_text):
    if not document_text or document_text.strip() == "":
        return {"error": "No content to analyze for risks."}
    
    # Define risk categories with keywords
    risk_categories = {
    "Regulatory Risks": [
        "compliance", "regulation", "violation", "legal obligation", "policy breach", 
        "statutory requirement", "government approval", "licensing", "audit", "penalty", 
        "reporting failure", "non-compliance", "due diligence", "risk assessment", 
        "legal proceedings"
    ],
    "Criminal Risks": [
        "crime", "offense", "penalty", "prosecution", "conviction", "imprisonment", 
        "fraud", "forgery", "embezzlement", "bribery", "money laundering", "extortion", 
        "smuggling", "harassment", "cybercrime", "homicide", "theft", "assault", 
        "domestic violence", "illegal possession"
    ],
    "Contractual Risks": [
        "contract breach", "liability", "damages", "negligence", "compensation", 
        "settlement", "plaintiff", "defendant", "arbitration", "mediation", "remedy", 
        "indemnity", "lawsuit", "decree", "jurisdiction", "litigation", "injunction", 
        "fiduciary duty", "legal notice"
    ],
    "Property Risks": [
        "ownership", "possession", "lease", "title dispute", "eviction", "trespassing", 
        "mortgage", "foreclosure", "inheritance", "zoning laws", "land acquisition", 
        "boundary dispute", "property rights", "transfer of property", "real estate fraud", 
        "encumbrance", "adverse possession", "land survey", "occupancy"
    ],
    "Financial Risks": [
        "tax evasion", "tax fraud", "financial misreporting", "audit", "revenue loss", 
        "bankruptcy", "debt recovery", "loan default", "insolvency", "foreclosure", 
        "credit risk", "monetary penalty", "securities fraud", "insider trading", 
        "banking regulations", "investment fraud", "fiscal policy violation", 
        "money laundering", "interest liability"
    ]
}

    # Normalize text for analysis
    text = document_text.lower()
    
    # Analyze each category and detect risks
    results = {}
    total_score = 0
    
    for category, keywords in risk_categories.items():
        category_score = 0
        keyword_hits = []
        
        for keyword in keywords:
            count = text.count(keyword)
            if count > 0:
                keyword_hits.append({"keyword": keyword, "count": count})
                category_score += count
        
        results[category] = {
            "score": category_score,
            "keywords": keyword_hits
        }
        total_score += category_score
    
    # Determine overall risk level
    risk_level = "Low"
    if total_score > 20:
        risk_level = "High"
    elif total_score > 10:
        risk_level = "Medium"
    
    # Prepare visualization data
    visualization_data = {
        "categories": list(risk_categories.keys()),
        "scores": [results[category]["score"] for category in risk_categories.keys()],
        "colors": []
    }
    
    # Assign colors based on score (for visualization)
    for score in visualization_data["scores"]:
        if score == 0:
            visualization_data["colors"].append("green")
        elif score < 3:
            visualization_data["colors"].append("yellow")
        elif score < 5:
            visualization_data["colors"].append("orange")
        else:
            visualization_data["colors"].append("red")
    
    # Get top risk sentences (optional)
    sentences = [s.strip() for s in text.replace(".", ". ").split(". ") if s.strip()]
    risk_sentences = []
    
    for sentence in sentences:
        for category, keywords in risk_categories.items():
            if any(keyword in sentence.lower() for keyword in keywords):
                risk_sentences.append(sentence)
                break
    
    # Compile final results
    final_results = {
        "risk_level": risk_level,
        "total_score": total_score,
        "category_results": results,
        "top_risk_sentences": risk_sentences[:5],  # Top 5 risky sentences
        "visualization_data": visualization_data  # Data ready for visualization
    }
    
    return final_results

def visualize_risks(risk_results):
    if "visualization_data" not in risk_results:
        return {"error": "No visualization data available"}

    viz_data = risk_results["visualization_data"]
    
    if not viz_data.get("categories") or not viz_data.get("scores"):
        return {"error": "Missing categories or scores in visualization data"}

    categories = viz_data["categories"]
    scores = viz_data["scores"]
    colors = viz_data["colors"]

    # Ensure valid data for visualization
    if not any(scores):  # If all scores are 0, display a message instead
        return {
            "bar_chart": None,
            "pie_chart": None
        }

    fig1, ax1 = plt.subplots(figsize=(10, 6))
    bars = ax1.bar(categories, scores, color=colors)
    ax1.set_xlabel('Risk Categories')
    ax1.set_ylabel('Risk Score')
    ax1.set_title('Risk Analysis by Category')
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')

    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1, f'{height}', ha='center', va='bottom')

    plt.tight_layout()

    fig2, ax2 = plt.subplots(figsize=(8, 8))
    non_zero_categories = [categories[i] for i, score in enumerate(scores) if score > 0]
    non_zero_scores = [score for score in scores if score > 0]

    if non_zero_scores:
        ax2.pie(non_zero_scores, labels=non_zero_categories, autopct='%1.1f%%', 
                startangle=90, colors=['yellow', 'orange', 'red', 'crimson', 'darkred'][:len(non_zero_scores)])
        ax2.set_title('Risk Distribution')
    else:
        ax2.text(0.5, 0.5, "No risks detected", ha='center', va='center')
        ax2.set_title('Risk Distribution (No Risks)')

    plt.tight_layout()

    return {"bar_chart": fig1, "pie_chart": fig2}
