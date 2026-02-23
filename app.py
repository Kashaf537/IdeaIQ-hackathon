import os
import json
import re
import faiss
import numpy as np
import gradio as gr
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import tempfile
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from groq import Groq

# ==========================
# OPTIONAL LIBRARIES
# ==========================
try:
    from fpdf import FPDF
    import kaleido
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    def random_projection(embeddings, n_components=3):
        np.random.seed(42)
        projection = np.random.randn(embeddings.shape[1], n_components)
        return np.dot(embeddings, projection)

# ==========================
# LOAD ENV & CONFIG
# ==========================
load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

BRAND_NAME = os.getenv("BRAND_NAME", "IdeaIQ")
LOGO_PATH = os.getenv("LOGO_PATH", "")

# ==========================
# LOAD DATASET
# ==========================
DATA_PATH = "cleaned_market_trends.json"  # adjust if needed

with open(DATA_PATH, "r", encoding="utf-8") as f:
    documents = json.load(f)

# ==========================
# EMBEDDINGS + FAISS
# ==========================
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

texts = [doc["content"] for doc in documents]
embeddings = embedding_model.encode(texts)
embeddings = np.array(embeddings).astype("float32")
faiss.normalize_L2(embeddings)

dimension = embeddings.shape[1]
faiss_index = faiss.IndexFlatIP(dimension)
faiss_index.add(embeddings)

# ==========================
# RETRIEVAL
# ==========================
def retrieve_context(user_idea, top_k=4):
    query_embedding = embedding_model.encode([user_idea])
    query_embedding = np.array(query_embedding).astype("float32")
    faiss.normalize_L2(query_embedding)

    scores, indices = faiss_index.search(query_embedding, top_k)

    retrieved_docs = []
    similarity_scores = []

    for score, idx in zip(scores[0], indices[0]):
        retrieved_docs.append(documents[idx])
        similarity_scores.append(float(score))

    avg_similarity = round(float(np.mean(similarity_scores)), 3)

    return retrieved_docs, avg_similarity

# ==========================
# JSON CLEANER
# ==========================
def extract_json(text):
    text = re.sub(r"```json", "", text)
    text = re.sub(r"```", "", text)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group())
    raise ValueError("Invalid JSON response")

# ==========================
# LLM ANALYSIS
# ==========================
SYSTEM_PROMPT = """
You are an AI Product Strategy Analyst.

Return ONLY valid JSON.
Do not wrap in markdown.
Strictly follow provided structure.
"""

def analyze_idea(user_idea):
    retrieved_docs, avg_similarity = retrieve_context(user_idea)

    context_text = "\n\n".join(
        [f"{doc['domain']} | {doc['year']} | {doc['content']}" for doc in retrieved_docs]
    )

    user_prompt = f"""
User Idea:
{user_idea}

Market Context:
{context_text}

Return strictly a JSON object with the following structure. The "market_analysis" list should contain one object per retrieved document (exactly {len(retrieved_docs)} items), with estimated numeric scores based on the document content and your market knowledge.

{{
  "idea": "{user_idea}",
  "scope": "Low | Medium | High",
  "feasibility": "Low | Medium | High",
  "estimated_timeline": "X months",
  "market_trend_summary": "...",
  "uniqueness_analysis": "...",
  "risks": ["Risk 1", "Risk 2"],
  "actionable_steps": ["Step 1", "Step 2"],
  "project_score": 0-10,
  "market_analysis": [
    {{
      "domain": "from document",
      "year": "from document",
      "category": "e.g., AI, Healthcare, Fintech",
      "region": "Global | North America | etc.",
      "growth_rate": 0-100,
      "innovation_score": 0-100,
      "market_potential": 0-100,
      "competition_level": 0-100
    }},
    ...
  ]
}}
"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )

    result = extract_json(response.choices[0].message.content)

    result.setdefault("scope", "Medium")
    result.setdefault("feasibility", "Medium")
    result.setdefault("estimated_timeline", "6 months")
    result.setdefault("market_trend_summary", "No summary")
    result.setdefault("uniqueness_analysis", "No analysis")
    result.setdefault("risks", ["Unknown"])
    result.setdefault("actionable_steps", ["Unknown"])
    result.setdefault("project_score", 5)
    result.setdefault("market_analysis", [])

    if not result["market_analysis"] and retrieved_docs:
        for doc in retrieved_docs:
            result["market_analysis"].append({
                "domain": doc["domain"],
                "year": doc["year"],
                "category": "Unknown",
                "region": "Global",
                "growth_rate": np.random.randint(30, 90),
                "innovation_score": np.random.randint(30, 90),
                "market_potential": np.random.randint(30, 90),
                "competition_level": np.random.randint(30, 90)
            })

    result["similarity_score"] = avg_similarity
    result["retrieved_docs"] = retrieved_docs

    return result

# ==========================
# VISUALIZATION FUNCTIONS
# ==========================
def create_gauge(result):
    project_score = result["project_score"]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=project_score,
        number={'suffix': "/10", 'font': {'size': 30}},
        title={'text': "Project Score", 'font': {'size': 20}},
        gauge={
            'axis': {'range': [0, 10], 'tickwidth': 1},
            'bar': {'color': "darkblue", 'thickness': 0.3},
            'bgcolor': "white",
            'borderwidth': 2,
            'steps': [
                {'range': [0, 3], 'color': '#ffcccc'},
                {'range': [3, 7], 'color': '#ffffcc'},
                {'range': [7, 10], 'color': '#ccffcc'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 7
            }
        }
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20),
                      paper_bgcolor='rgba(0,0,0,0)')
    return fig

def create_radial_graph(result, visible_nodes=8):
    idea = result["idea"]
    project_score = result["project_score"]
    feasibility = result["feasibility"]
    similarity = result["similarity_score"]
    scope = result["scope"]
    timeline = result["estimated_timeline"]

    score_color = f"rgb({255-int(project_score*25)}, {int(project_score*25)}, 0)"
    feasibility_color = {"High": "#00cc96", "Medium": "#ffa15e", "Low": "#ef553b"}.get(feasibility, "#636efa")

    nodes = [
        idea,
        f"Scope: {scope}",
        f"Feasibility: {feasibility}",
        f"Timeline: {timeline}",
        "Risks",
        "Actionable Steps",
        "Uniqueness",
        f"Similarity: {similarity}"
    ]

    angle = np.linspace(0, 2*np.pi, len(nodes))
    radius = 2.5
    x = [0] + list(radius * np.cos(angle[1:]))
    y = [0] + list(radius * np.sin(angle[1:]))

    colors = [
        score_color,
        "#1f77b4",
        feasibility_color,
        "#ff7f0e",
        "#d62728",
        "#9467bd",
        "#17becf",
        "#bcbd22"
    ]
    sizes = [60, 35, 35, 35, 40, 40, 35, 35]

    hover_texts = [
        f"<b>Idea</b><br>{idea}<br><b>Score: {project_score}/10</b>",
        f"<b>Scope</b><br>{scope}",
        f"<b>Feasibility</b><br>{feasibility}",
        f"<b>Timeline</b><br>{timeline}",
        f"<b>Risks</b><br>" + "<br>".join(result["risks"]),
        f"<b>Actionable Steps</b><br>" + "<br>".join(result["actionable_steps"]),
        f"<b>Uniqueness</b><br>{result['uniqueness_analysis']}",
        f"<b>Similarity Score</b><br>{similarity}"
    ]

    visible = min(visible_nodes, len(nodes))
    visible_x = x[:visible]
    visible_y = y[:visible]
    visible_text = nodes[:visible]
    visible_hover = hover_texts[:visible]
    visible_colors = colors[:visible]
    visible_sizes = sizes[:visible]

    edge_x, edge_y = [], []
    for i in range(1, visible):
        edge_x += [x[0], x[i], None]
        edge_y += [y[0], y[i], None]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode='lines',
        line=dict(width=1.5, color='#888'),
        hoverinfo='none'
    ))

    fig.add_trace(go.Scatter(
        x=visible_x, y=visible_y,
        mode='markers+text',
        text=visible_text,
        textposition='bottom center',
        hovertext=visible_hover,
        hoverinfo='text',
        marker=dict(
            size=visible_sizes,
            color=visible_colors,
            line=dict(width=2, color='white')
        ),
        textfont=dict(size=12, color='black')
    ))

    fig.update_layout(
        title=dict(text="🚀 Strategic Opportunity Map", font=dict(size=20, family='Arial Black')),
        showlegend=False,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        height=550,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=80, b=20)
    )
    return fig

def create_3d_market(result):
    retrieved_docs = result.get("retrieved_docs", [])
    idea = result["idea"]

    if not retrieved_docs:
        return go.Figure()

    doc_texts = [doc["content"] for doc in retrieved_docs]
    all_texts = [idea] + doc_texts
    all_embeddings = embedding_model.encode(all_texts)

    if SKLEARN_AVAILABLE:
        pca = PCA(n_components=3)
        coords = pca.fit_transform(all_embeddings)
    else:
        coords = random_projection(all_embeddings, 3)

    df = pd.DataFrame({
        'x': coords[:,0],
        'y': coords[:,1],
        'z': coords[:,2],
        'label': ['Your Idea'] + [f"{doc['domain']} ({doc['year']})" for doc in retrieved_docs],
        'size': [20] + [15]*len(retrieved_docs),
        'color': ['red'] + ['blue']*len(retrieved_docs),
        'hover': [f"<b>Your Idea</b><br>{idea}"] + 
                 [f"<b>{doc['domain']} ({doc['year']})</b><br>{doc['content'][:100]}..." for doc in retrieved_docs]
    })

    fig = px.scatter_3d(df, x='x', y='y', z='z', text='label', size='size', color='color',
                         hover_data={'hover': True, 'x': False, 'y': False, 'z': False, 'color': False, 'size': False},
                         title="🧠 Market Landscape (3D Similarity Space)")
    fig.update_traces(marker=dict(line=dict(width=2, color='white')),
                      textposition='top center')
    fig.update_layout(height=550, showlegend=False)
    return fig

def create_kpi_panel(df):
    fig = go.Figure()
    metrics = {
        "Avg Growth": df["growth_rate"].mean(),
        "Innovation Strength": df["innovation_score"].mean(),
        "Market Potential": df["market_potential"].mean(),
        "Strategic Advantage": (df["innovation_score"] + df["market_potential"] - df["competition_level"]).mean()
    }

    fig.add_trace(go.Indicator(
        mode="number",
        value=round(metrics["Avg Growth"], 2),
        title={"text": "Avg Growth"},
        domain={'x': [0, 0.45], 'y': [0.55, 1]},
        number={"font": {"size": 36}}
    ))
    fig.add_trace(go.Indicator(
        mode="number",
        value=round(metrics["Innovation Strength"], 2),
        title={"text": "Innovation Strength"},
        domain={'x': [0.55, 1], 'y': [0.55, 1]},
        number={"font": {"size": 36}}
    ))
    fig.add_trace(go.Indicator(
        mode="number",
        value=round(metrics["Market Potential"], 2),
        title={"text": "Market Potential"},
        domain={'x': [0, 0.45], 'y': [0, 0.45]},
        number={"font": {"size": 36}}
    ))
    fig.add_trace(go.Indicator(
        mode="number",
        value=round(metrics["Strategic Advantage"], 2),
        title={"text": "Strategic Advantage"},
        domain={'x': [0.55, 1], 'y': [0, 0.45]},
        number={"font": {"size": 36}}
    ))

    fig.update_layout(
        template="plotly_dark",
        height=320,
        margin=dict(t=20, b=20, l=20, r=20)
    )
    return fig

def create_positioning_matrix(df):
    fig = px.scatter(
        df,
        x="competition_level",
        y="innovation_score",
        size="market_potential",
        color="growth_rate",
        hover_data=["domain","category","region","year"],
        title="Strategic Positioning Matrix",
        size_max=50,
        color_continuous_scale="Turbo"
    )
    fig.update_traces(
        marker=dict(line=dict(width=1, color='white')),
        hovertemplate="<b>%{customdata[0]}</b><br>Category: %{customdata[1]}<br>Region: %{customdata[2]}<br>Year: %{customdata[3]}<br>Growth: %{color:.1f}<extra></extra>"
    )
    fig.update_layout(
        template="plotly_dark",
        height=400,
        hovermode="closest",
        transition={"duration": 500}
    )
    return fig

def create_growth_by_year(df, selected_year):
    if "year" not in df.columns or df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data available", showarrow=False)
        fig.update_layout(template="plotly_dark", height=400)
        return fig

    # Convert year column to string to match slider value
    df = df.copy()
    df['year'] = df['year'].astype(str)

    df_filtered = df[df["year"] == selected_year].copy()
    if df_filtered.empty:
        fig = go.Figure()
        fig.add_annotation(text=f"No data for year {selected_year}", showarrow=False)
        fig.update_layout(template="plotly_dark", height=400)
        return fig

    fig = px.bar(
        df_filtered,
        x="domain",
        y="growth_rate",
        color="growth_rate",
        title=f"Domain Growth - {selected_year}",
        color_continuous_scale="Viridis",
        range_y=[0, 100],
        hover_data=["category", "region"]
    )
    fig.update_traces(
        hovertemplate="<b>%{x}</b><br>Growth: %{y:.1f}<br>Category: %{customdata[0]}<br>Region: %{customdata[1]}<extra></extra>"
    )
    fig.update_layout(
        template="plotly_dark",
        height=400,
        xaxis_title="Domain",
        yaxis_title="Growth Rate (%)"
    )
    return fig

def create_heatmap(df):
    if "year" not in df.columns:
        return go.Figure()
    pivot = df.pivot_table(values="market_potential", index="domain", columns="year")
    fig = px.imshow(
        pivot,
        color_continuous_scale="Magma",
        title="Market Potential Heatmap",
        aspect="auto",
        labels=dict(x="Year", y="Domain", color="Potential")
    )
    fig.update_layout(template="plotly_dark", height=400, transition={"duration": 500})
    fig.update_traces(hovertemplate="Year: %{x}<br>Domain: %{y}<br>Potential: %{z:.1f}<extra></extra>")
    return fig

def create_opportunity_radar(df):
    if df.empty:
        return go.Figure()
    df["advantage"] = df["innovation_score"] + df["market_potential"] - df["competition_level"]
    top = df.sort_values("advantage", ascending=False).iloc[0]
    categories = ["Growth", "Innovation", "Market", "Competitive Advantage"]
    values = [
        top["growth_rate"],
        top["innovation_score"],
        top["market_potential"],
        100 - top["competition_level"]
    ]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill="toself",
        marker=dict(color='rgba(102, 126, 234, 0.8)'),
        hovertemplate="%{theta}: %{r:.1f}<extra></extra>"
    ))
    fig.update_layout(
        template="plotly_dark",
        polar=dict(radialaxis=dict(range=[0,100], visible=True)),
        title=f"Top Opportunity: {top['domain']}",
        height=400,
        transition={"duration": 500}
    )
    return fig

# ==========================
# PDF GENERATION (FIXED)
# ==========================
REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

def generate_pdf(result, radial_fig):
    if not PDF_AVAILABLE:
        error_path = os.path.join(REPORTS_DIR, "pdf_error.txt")
        with open(error_path, "w") as f:
            f.write("PDF generation is not available because required libraries (fpdf2, kaleido) are missing.\nPlease install them.")
        return error_path

    try:
        # Test if kaleido can export (will fail on HF if chromium missing)
        try:
            _ = radial_fig.to_image(format="png")
        except Exception as e:
            error_path = os.path.join(REPORTS_DIR, "export_error.txt")
            with open(error_path, "w") as f:
                f.write(f"Failed to export plot image: {str(e)}\nThis may be due to missing Chromium.")
            return error_path

        pdf = FPDF()
        pdf.add_page()

        if LOGO_PATH and os.path.exists(LOGO_PATH):
            pdf.image(LOGO_PATH, x=10, y=8, w=30)
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, BRAND_NAME, ln=1, align='C')
        pdf.ln(10)

        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "Executive Summary", ln=1)
        pdf.set_font("Arial", '', 10)
        summary = f"""
Project Score: {result['project_score']}/10
Feasibility: {result['feasibility']}
Timeline: {result['estimated_timeline']}
Market Similarity: {result['similarity_score']}

Market Trend Summary:
{result['market_trend_summary']}

Uniqueness Analysis:
{result['uniqueness_analysis']}

Top Risks:
{chr(10).join(['- ' + r for r in result['risks']])}

Actionable Steps:
{chr(10).join(['- ' + s for s in result['actionable_steps']])}
"""
        pdf.multi_cell(0, 5, summary)
        pdf.ln(5)

        # Add radial graph image
        img_bytes = radial_fig.to_image(format="png")
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_img:
            tmp_img.write(img_bytes)
            img_path = tmp_img.name
        pdf.image(img_path, w=180)
        os.unlink(img_path)
        pdf.ln(5)

        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "Structured Report (JSON)", ln=1)
        pdf.set_font("Courier", '', 8)
        pdf.multi_cell(0, 4, json.dumps(result, indent=2))

        # Save PDF with timestamp
        pdf_filename = f"report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        pdf_path = os.path.join(REPORTS_DIR, pdf_filename)
        pdf.output(pdf_path)

        if os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 0:
            return pdf_path
        else:
            error_path = os.path.join(REPORTS_DIR, "empty_pdf.txt")
            with open(error_path, "w") as f:
                f.write("PDF file was created but appears empty.")
            return error_path
    except Exception as e:
        print(f"PDF generation error: {e}")
        error_path = os.path.join(REPORTS_DIR, "pdf_error.txt")
        with open(error_path, "w") as f:
            f.write(f"An error occurred during PDF generation: {str(e)}")
        return error_path

def prepare_pdf(result, radial_fig):
    if not result or not radial_fig:
        error_path = os.path.join(REPORTS_DIR, "no_data.txt")
        with open(error_path, "w") as f:
            f.write("No data available for PDF generation.")
        return error_path
    return generate_pdf(result, radial_fig)

# ==========================
# UPDATE FUNCTIONS
# ==========================
def update_radial_and_state(result, step):
    if not result:
        return go.Figure(), go.Figure()
    new_fig = create_radial_graph(result, step)
    return new_fig, new_fig

def update_live_score(result_state):
    if result_state:
        new_score = result_state.get("similarity_score", 0.5)
        variation = np.random.uniform(-0.05, 0.05)
        new_score = round(max(0, min(1, new_score + variation)), 3)
        result_state["similarity_score"] = new_score
        return f"<span class='live-badge'>Live Similarity: {new_score}</span>"
    return "<span class='live-badge'>Live Similarity: N/A</span>"

def increment_step(current_step):
    return min(current_step + 1, 8)

def update_growth_by_year(result_state, selected_year):
    if not result_state:
        return go.Figure()
    df = pd.DataFrame(result_state.get("market_analysis", []))
    for col in ["growth_rate","innovation_score","market_potential","competition_level"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df.fillna(0, inplace=True)
    return create_growth_by_year(df, selected_year)

# ==========================
# FULL PIPELINE
# ==========================
def full_pipeline(user_input):
    try:
        result = analyze_idea(user_input)

        fig_radial = create_radial_graph(result, visible_nodes=1)
        fig_3d = create_3d_market(result)
        fig_gauge = create_gauge(result)

        df = pd.DataFrame(result["market_analysis"])
        for col in ["growth_rate","innovation_score","market_potential","competition_level"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df.fillna(0, inplace=True)

        years = []
        if "year" in df.columns:
            years = sorted(df["year"].unique())
            years = [str(y) for y in years]

        if df.empty:
            fig_kpi = go.Figure()
            fig_matrix = go.Figure()
            fig_growth = create_growth_by_year(df, None)
            fig_heat = go.Figure()
            fig_radar_opp = go.Figure()
        else:
            fig_kpi = create_kpi_panel(df)
            fig_matrix = create_positioning_matrix(df)
            initial_year = str(df["year"].iloc[0]) if "year" in df.columns else None
            fig_growth = create_growth_by_year(df, initial_year)
            fig_heat = create_heatmap(df)
            fig_radar_opp = create_opportunity_radar(df)

        summary_md = f"""
### 📊 Key Metrics
- **Project Score**: **{result['project_score']}/10**
- **Feasibility**: **{result['feasibility']}**
- **Timeline**: **{result['estimated_timeline']}**
- **Market Similarity**: **{result['similarity_score']}**

### 📈 Market Trend Summary
{result['market_trend_summary']}

### 💡 Uniqueness Analysis
{result['uniqueness_analysis']}

### ⚠️ Top Risks
{chr(10).join(['- ' + r for r in result['risks']])}

### ✅ Actionable Steps
{chr(10).join(['- ' + s for s in result['actionable_steps']])}
"""

        confetti_html = ""
        if result['project_score'] > 8:
            confetti_html = """
            <script src="https://cdn.jsdelivr.net/npm/canvas-confetti@1"></script>
            <script>
                confetti({particleCount: 100, spread: 70, origin: { y: 0.6 }});
            </script>
            """

        return (summary_md, json.dumps(result, indent=4),
                fig_radial, fig_3d, fig_gauge,
                fig_kpi, fig_matrix, fig_growth, fig_heat, fig_radar_opp,
                result, fig_radial, confetti_html, years)
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        empty_fig = go.Figure()
        return (error_msg, "{}", empty_fig, empty_fig, empty_fig,
                empty_fig, empty_fig, empty_fig, empty_fig, empty_fig,
                {}, empty_fig, "", [])

# ==========================
# ULTRA‑PREMIUM RESPONSIVE UI
# ==========================
with gr.Blocks(theme=gr.themes.Soft(), css="") as app:
    gr.HTML("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Inter', sans-serif;
            background: #f8fafc;  /* Soft, professional light gray */
            min-height: 100vh;
        }

        .gradio-container {
            max-width: 1600px;
            margin: 1.5rem auto;
            padding: 1.5rem;
            background: #ffffff;
            border-radius: 2rem;
            box-shadow: 0 20px 40px -10px rgba(0,0,0,0.1);
        }

        /* Animated gradient header */
        .premium-header {
            background: linear-gradient(-45deg, #1e3c72, #2a5298, #1e3c72, #162b4c);
            background-size: 400% 400%;
            animation: gradient 15s ease infinite;
            padding: 2.5rem 2rem;
            border-radius: 1.5rem;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
            box-shadow: 0 10px 25px -5px rgba(0,0,0,0.2);
            position: relative;
            overflow: hidden;
        }

        .premium-header::before {
            content: "";
            position: absolute;
            top: -50%;
            right: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, rgba(255,255,255,0.15) 0%, transparent 70%);
            animation: rotate 20s linear infinite;
        }

        @keyframes gradient {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        @keyframes rotate {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
        }

        .premium-header h1 {
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
            letter-spacing: -0.02em;
            position: relative;
            z-index: 2;
        }

        .premium-header p {
            font-size: 1.1rem;
            opacity: 0.9;
            font-weight: 300;
            position: relative;
            z-index: 2;
        }

        /* Input styling – clean, modern */
        .premium-input textarea {
            border: 1px solid #e2e8f0 !important;
            border-radius: 1rem !important;
            padding: 0.75rem 1rem !important;
            font-size: 1rem !important;
            background: white !important;
            box-shadow: 0 2px 4px rgba(0,0,0,0.02) !important;
            transition: all 0.2s !important;
            width: 100% !important;
        }

        .premium-input textarea:focus {
            border-color: #3182ce !important;
            box-shadow: 0 0 0 3px rgba(49,130,206,0.1) !important;
            outline: none !important;
        }

        /* Buttons – modern, with hover effects */
        .btn-primary {
            background: #3182ce;
            color: white;
            border: none;
            padding: 0.6rem 1.5rem;
            border-radius: 2rem;
            font-weight: 500;
            font-size: 1rem;
            cursor: pointer;
            transition: all 0.2s;
            box-shadow: 0 4px 6px -1px rgba(49,130,206,0.3);
            width: 100%;
            display: inline-flex;
            align-items: center;
            justify-content: center;
        }

        .btn-primary:hover {
            background: #2c5282;
            transform: translateY(-1px);
            box-shadow: 0 6px 8px -1px rgba(49,130,206,0.4);
        }

        .btn-secondary {
            background: white;
            color: #1a202c;
            border: 1px solid #cbd5e0;
            padding: 0.5rem 1.2rem;
            border-radius: 2rem;
            font-weight: 500;
            font-size: 0.95rem;
            cursor: pointer;
            transition: all 0.2s;
            display: inline-flex;
            align-items: center;
            justify-content: center;
        }

        .btn-secondary:hover {
            background: #f7fafc;
            border-color: #a0aec0;
        }

        /* Cards – glass effect */
        .glass-card {
            background: rgba(255,255,255,0.8);
            backdrop-filter: blur(10px);
            border-radius: 1.5rem;
            padding: 1.5rem;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
            border: 1px solid rgba(255,255,255,0.5);
            height: fit-content;
        }

        /* Plot cards */
        .plot-card {
            background: white;
            border-radius: 1.2rem;
            padding: 1rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            border: 1px solid #edf2f7;
            transition: all 0.2s;
            margin-bottom: 1rem;
        }

        .plot-card:hover {
            box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1);
        }

        /* Live badge */
        .live-badge {
            background: #3182ce;
            color: white;
            padding: 0.4rem 1.2rem;
            border-radius: 2rem;
            font-size: 0.9rem;
            font-weight: 500;
            display: inline-block;
            box-shadow: 0 2px 4px rgba(49,130,206,0.2);
            margin: 0.5rem 0;
        }

        /* Slider styling */
        .premium-slider {
            background: white;
            border-radius: 1rem;
            padding: 0.5rem 1rem;
            border: 1px solid #e2e8f0;
        }

        /* Responsive design */
        @media (max-width: 768px) {
            .gradio-container {
                padding: 1rem;
                margin: 0.5rem;
                border-radius: 1.5rem;
            }
            .premium-header h1 {
                font-size: 1.8rem;
            }
            .premium-header p {
                font-size: 0.95rem;
            }
            .btn-primary, .btn-secondary {
                width: 100%;
                margin-top: 0.5rem;
            }
            /* Stack columns on mobile */
            .gradio-row {
                flex-direction: column !important;
            }
            .gradio-column {
                width: 100% !important;
                min-width: 100% !important;
            }
        }

        /* Larger screens: allow more horizontal space */
        @media (min-width: 1600px) {
            .gradio-container {
                max-width: 1800px;
                padding: 2.5rem;
            }
            .premium-header h1 {
                font-size: 3rem;
            }
        }

        /* Fade-in animation */
        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .fade-in {
            animation: fadeInUp 0.5s ease-out;
        }
    </style>
    """)

    # ==========================
    # HEADER
    # ==========================
    gr.HTML(f"""
    <div class="premium-header fade-in">
        <h1>🚀 {BRAND_NAME}</h1>
        <p>AI-Powered Market Intelligence • Real-Time Analysis • Strategic Insights</p>
    </div>
    """)

    # ==========================
    # INPUT ROW (NO VOICE)
    # ==========================
    with gr.Row(elem_classes="gradio-row"):
        with gr.Column(scale=4, elem_classes="gradio-column"):
            idea_input = gr.Textbox(
                placeholder="Enter your startup idea... e.g., AI platform for personalized mental health coaching",
                lines=2,
                container=False,
                elem_classes="premium-input"
            )
        with gr.Column(scale=1, min_width=120, elem_classes="gradio-column"):
            analyze_btn = gr.Button("Analyze", elem_classes="btn-primary")

    # ==========================
    # STATES
    # ==========================
    result_state = gr.State()
    radial_fig_state = gr.State()
    years_state = gr.State()

    # Live similarity badge
    live_display = gr.HTML('<span class="live-badge">Live Similarity: N/A</span>')

    # ==========================
    # SUMMARY & JSON
    # ==========================
    with gr.Row(elem_classes="gradio-row"):
        with gr.Column(scale=1, elem_classes="glass-card fade-in gradio-column"):
            summary_output = gr.Markdown()
        with gr.Column(scale=1, elem_classes="glass-card fade-in gradio-column"):
            json_output = gr.Code(label="📄 Structured Report", language="json", lines=10)

    # ==========================
    # PLOT ROWS
    # ==========================
    # Row 1
    with gr.Row(elem_classes="gradio-row"):
        with gr.Column(elem_classes="plot-card gradio-column"):
            radial_output = gr.Plot(label="🗺️ Strategic Opportunity Map")
        with gr.Column(elem_classes="plot-card gradio-column"):
            market_output = gr.Plot(label="🧠 3D Market Landscape")
        with gr.Column(elem_classes="plot-card gradio-column"):
            gauge_output = gr.Plot(label="🎯 Project Score")

    # Row 2
    with gr.Row(elem_classes="gradio-row"):
        with gr.Column(elem_classes="plot-card gradio-column"):
            kpi_output = gr.Plot(label="📊 KPI Panel")
        with gr.Column(elem_classes="plot-card gradio-column"):
            matrix_output = gr.Plot(label="🎯 Positioning Matrix")

    # Row 3 (Growth with year slider)
    with gr.Row(elem_classes="gradio-row"):
        with gr.Column(elem_classes="plot-card gradio-column"):
            year_slider = gr.Slider(
                minimum=0, maximum=0, step=1, value=0,
                label="Select Year", visible=False,
                elem_classes="premium-slider"
            )
            growth_output = gr.Plot(label="📈 Growth by Year")

    # Row 4
    with gr.Row(elem_classes="gradio-row"):
        with gr.Column(elem_classes="plot-card gradio-column"):
            heat_output = gr.Plot(label="🔥 Market Heatmap")
        with gr.Column(elem_classes="plot-card gradio-column"):
            radar_opp_output = gr.Plot(label="⭐ Opportunity Radar")

    # ==========================
    # CONTROLS
    # ==========================
    with gr.Row(elem_classes="glass-card fade-in gradio-row"):
        step_slider = gr.Slider(minimum=1, maximum=8, step=1, value=1, label="Reveal Steps", elem_classes="premium-slider")
        reveal_btn = gr.Button("Reveal Next Step", elem_classes="btn-secondary")
        live_btn = gr.Button("Refresh Live Similarity", elem_classes="btn-secondary")
        pdf_btn = gr.Button("📥 Download PDF Report", elem_classes="btn-primary")
    pdf_output = gr.File(label="Download PDF", visible=True)
    confetti_html = gr.HTML()

    # ==========================
    # EVENT HANDLERS
    # ==========================
    def update_after_analysis(summary, json, radial, market, gauge, 
                              kpi, matrix, growth, heat, radar_opp, 
                              result, radial_fig, confetti, years):
        if years and len(years) > 0:
            return (summary, json, radial, market, gauge,
                    kpi, matrix, growth, heat, radar_opp,
                    result, radial_fig, confetti,
                    gr.update(minimum=0, maximum=len(years)-1, step=1, 
                             value=0, visible=True, label=f"Year: {years[0]}"),
                    years)
        else:
            return (summary, json, radial, market, gauge,
                    kpi, matrix, growth, heat, radar_opp,
                    result, radial_fig, confetti,
                    gr.update(visible=False), years)

    analyze_btn.click(
        fn=full_pipeline,
        inputs=idea_input,
        outputs=[summary_output, json_output,
                 radial_output, market_output, gauge_output,
                 kpi_output, matrix_output, growth_output, heat_output, radar_opp_output,
                 result_state, radial_fig_state, confetti_html, years_state]
    ).then(
        fn=update_after_analysis,
        inputs=[summary_output, json_output,
                radial_output, market_output, gauge_output,
                kpi_output, matrix_output, growth_output, heat_output, radar_opp_output,
                result_state, radial_fig_state, confetti_html, years_state],
        outputs=[summary_output, json_output,
                 radial_output, market_output, gauge_output,
                 kpi_output, matrix_output, growth_output, heat_output, radar_opp_output,
                 result_state, radial_fig_state, confetti_html, year_slider, years_state]
    ).then(
        fn=lambda rs: f"<span class='live-badge'>Live Similarity: {rs.get('similarity_score', 'N/A')}</span>",
        inputs=result_state,
        outputs=live_display
    )

    def on_year_change(result_state, slider_val, years_state):
        if years_state and slider_val < len(years_state):
            selected_year = years_state[slider_val]
            return update_growth_by_year(result_state, selected_year)
        return go.Figure()

    year_slider.change(
        fn=on_year_change,
        inputs=[result_state, year_slider, years_state],
        outputs=growth_output
    )

    def update_slider_label(slider_val, years_state):
        if years_state and slider_val < len(years_state):
            return gr.update(label=f"Year: {years_state[slider_val]}")
        return gr.update(label="Select Year")

    year_slider.change(
        fn=update_slider_label,
        inputs=[year_slider, years_state],
        outputs=year_slider
    )

    step_slider.change(
        fn=update_radial_and_state,
        inputs=[result_state, step_slider],
        outputs=[radial_output, radial_fig_state]
    )

    reveal_btn.click(
        fn=increment_step,
        inputs=step_slider,
        outputs=step_slider
    ).then(
        fn=update_radial_and_state,
        inputs=[result_state, step_slider],
        outputs=[radial_output, radial_fig_state]
    )

    live_btn.click(
        fn=update_live_score,
        inputs=result_state,
        outputs=live_display
    )

    pdf_btn.click(
        fn=prepare_pdf,
        inputs=[result_state, radial_fig_state],
        outputs=pdf_output
    )

    # ==========================
    # FOOTER
    # ==========================
    gr.HTML("""
    <div style="text-align: center; margin-top: 3rem; padding: 1rem; color: #718096; font-size: 0.9rem;">
        <p>Powered by Groq LLama 3.3 • FAISS • Sentence‑Transformers</p>
    </div>
    """)

if __name__ == "__main__":
    app.launch()
# import os
# import json
# import re
# import faiss
# import numpy as np
# import gradio as gr
# import plotly.graph_objects as go
# import plotly.express as px
# import pandas as pd
# import tempfile
# from io import BytesIO
# from dotenv import load_dotenv
# from sentence_transformers import SentenceTransformer
# from groq import Groq

# # Optional: for PDF export
# try:
#     from fpdf import FPDF
#     import kaleido
#     PDF_AVAILABLE = True
# except ImportError:
#     PDF_AVAILABLE = False

# # Optional: for 3D PCA
# try:
#     from sklearn.decomposition import PCA
#     SKLEARN_AVAILABLE = True
# except ImportError:
#     SKLEARN_AVAILABLE = False
#     def random_projection(embeddings, n_components=3):
#         np.random.seed(42)
#         projection = np.random.randn(embeddings.shape[1], n_components)
#         return np.dot(embeddings, projection)

# # ==========================
# # LOAD ENV & CONFIG
# # ==========================
# load_dotenv()
# groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# # Custom branding
# BRAND_NAME = os.getenv("BRAND_NAME", "IdeaIQ")
# LOGO_PATH = os.getenv("LOGO_PATH", "")

# # ==========================
# # LOAD DATASET
# # ==========================
# DATA_PATH = "datasets/processed/cleaned_market_trends.json"

# with open(DATA_PATH, "r", encoding="utf-8") as f:
#     documents = json.load(f)

# # ==========================
# # EMBEDDINGS + FAISS
# # ==========================
# embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# texts = [doc["content"] for doc in documents]
# embeddings = embedding_model.encode(texts)
# embeddings = np.array(embeddings).astype("float32")
# faiss.normalize_L2(embeddings)

# dimension = embeddings.shape[1]
# faiss_index = faiss.IndexFlatIP(dimension)
# faiss_index.add(embeddings)

# # ==========================
# # RETRIEVAL
# # ==========================
# def retrieve_context(user_idea, top_k=4):
#     query_embedding = embedding_model.encode([user_idea])
#     query_embedding = np.array(query_embedding).astype("float32")
#     faiss.normalize_L2(query_embedding)

#     scores, indices = faiss_index.search(query_embedding, top_k)

#     retrieved_docs = []
#     similarity_scores = []

#     for score, idx in zip(scores[0], indices[0]):
#         retrieved_docs.append(documents[idx])
#         similarity_scores.append(float(score))

#     avg_similarity = round(float(np.mean(similarity_scores)), 3)

#     return retrieved_docs, avg_similarity

# # ==========================
# # JSON CLEANER
# # ==========================
# def extract_json(text):
#     text = re.sub(r"```json", "", text)
#     text = re.sub(r"```", "", text)
#     match = re.search(r"\{.*\}", text, re.DOTALL)
#     if match:
#         return json.loads(match.group())
#     raise ValueError("Invalid JSON response")

# # ==========================
# # LLM ANALYSIS
# # ==========================
# SYSTEM_PROMPT = """
# You are an AI Product Strategy Analyst.

# Return ONLY valid JSON.
# Do not wrap in markdown.
# Strictly follow provided structure.
# """

# def analyze_idea(user_idea):
#     retrieved_docs, avg_similarity = retrieve_context(user_idea)

#     context_text = "\n\n".join(
#         [f"{doc['domain']} | {doc['year']} | {doc['content']}" for doc in retrieved_docs]
#     )

#     user_prompt = f"""
# User Idea:
# {user_idea}

# Market Context:
# {context_text}

# Return strictly a JSON object with the following structure. The "market_analysis" list should contain one object per retrieved document (exactly {len(retrieved_docs)} items), with estimated numeric scores based on the document content and your market knowledge.

# {{
#   "idea": "{user_idea}",
#   "scope": "Low | Medium | High",
#   "feasibility": "Low | Medium | High",
#   "estimated_timeline": "X months",
#   "market_trend_summary": "...",
#   "uniqueness_analysis": "...",
#   "risks": ["Risk 1", "Risk 2"],
#   "actionable_steps": ["Step 1", "Step 2"],
#   "project_score": 0-10,
#   "market_analysis": [
#     {{
#       "domain": "from document",
#       "year": "from document",
#       "category": "e.g., AI, Healthcare, Fintech",
#       "region": "Global | North America | etc.",
#       "growth_rate": 0-100,
#       "innovation_score": 0-100,
#       "market_potential": 0-100,
#       "competition_level": 0-100
#     }},
#     ...
#   ]
# }}
# """

#     response = groq_client.chat.completions.create(
#         model="llama-3.3-70b-versatile",
#         messages=[
#             {"role": "system", "content": SYSTEM_PROMPT},
#             {"role": "user", "content": user_prompt}
#         ],
#         temperature=0
#     )

#     result = extract_json(response.choices[0].message.content)

#     result.setdefault("scope", "Medium")
#     result.setdefault("feasibility", "Medium")
#     result.setdefault("estimated_timeline", "6 months")
#     result.setdefault("market_trend_summary", "No summary")
#     result.setdefault("uniqueness_analysis", "No analysis")
#     result.setdefault("risks", ["Unknown"])
#     result.setdefault("actionable_steps", ["Unknown"])
#     result.setdefault("project_score", 5)
#     result.setdefault("market_analysis", [])

#     if not result["market_analysis"] and retrieved_docs:
#         for doc in retrieved_docs:
#             result["market_analysis"].append({
#                 "domain": doc["domain"],
#                 "year": doc["year"],
#                 "category": "Unknown",
#                 "region": "Global",
#                 "growth_rate": np.random.randint(30, 90),
#                 "innovation_score": np.random.randint(30, 90),
#                 "market_potential": np.random.randint(30, 90),
#                 "competition_level": np.random.randint(30, 90)
#             })

#     result["similarity_score"] = avg_similarity
#     result["retrieved_docs"] = retrieved_docs

#     return result

# # ==========================
# # GAUGE FUNCTION
# # ==========================
# def create_gauge(result):
#     project_score = result["project_score"]
#     fig = go.Figure(go.Indicator(
#         mode="gauge+number",
#         value=project_score,
#         number={'suffix': "/10", 'font': {'size': 30}},
#         title={'text': "Project Score", 'font': {'size': 20}},
#         gauge={
#             'axis': {'range': [0, 10], 'tickwidth': 1},
#             'bar': {'color': "darkblue", 'thickness': 0.3},
#             'bgcolor': "white",
#             'borderwidth': 2,
#             'steps': [
#                 {'range': [0, 3], 'color': '#ffcccc'},
#                 {'range': [3, 7], 'color': '#ffffcc'},
#                 {'range': [7, 10], 'color': '#ccffcc'}
#             ],
#             'threshold': {
#                 'line': {'color': "red", 'width': 4},
#                 'thickness': 0.75,
#                 'value': 7
#             }
#         }
#     ))
#     fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20),
#                       paper_bgcolor='rgba(0,0,0,0)')
#     return fig

# # ==========================
# # RADIAL GRAPH
# # ==========================
# def create_radial_graph(result, visible_nodes=8):
#     idea = result["idea"]
#     project_score = result["project_score"]
#     feasibility = result["feasibility"]
#     similarity = result["similarity_score"]
#     scope = result["scope"]
#     timeline = result["estimated_timeline"]

#     score_color = f"rgb({255-int(project_score*25)}, {int(project_score*25)}, 0)"
#     feasibility_color = {"High": "#00cc96", "Medium": "#ffa15e", "Low": "#ef553b"}.get(feasibility, "#636efa")

#     nodes = [
#         idea,
#         f"Scope: {scope}",
#         f"Feasibility: {feasibility}",
#         f"Timeline: {timeline}",
#         "Risks",
#         "Actionable Steps",
#         "Uniqueness",
#         f"Similarity: {similarity}"
#     ]

#     angle = np.linspace(0, 2*np.pi, len(nodes))
#     radius = 2.5
#     x = [0] + list(radius * np.cos(angle[1:]))
#     y = [0] + list(radius * np.sin(angle[1:]))

#     colors = [
#         score_color,
#         "#1f77b4",
#         feasibility_color,
#         "#ff7f0e",
#         "#d62728",
#         "#9467bd",
#         "#17becf",
#         "#bcbd22"
#     ]
#     sizes = [60, 35, 35, 35, 40, 40, 35, 35]

#     hover_texts = [
#         f"<b>Idea</b><br>{idea}<br><b>Score: {project_score}/10</b>",
#         f"<b>Scope</b><br>{scope}",
#         f"<b>Feasibility</b><br>{feasibility}",
#         f"<b>Timeline</b><br>{timeline}",
#         f"<b>Risks</b><br>" + "<br>".join(result["risks"]),
#         f"<b>Actionable Steps</b><br>" + "<br>".join(result["actionable_steps"]),
#         f"<b>Uniqueness</b><br>{result['uniqueness_analysis']}",
#         f"<b>Similarity Score</b><br>{similarity}"
#     ]

#     visible = min(visible_nodes, len(nodes))
#     visible_x = x[:visible]
#     visible_y = y[:visible]
#     visible_text = nodes[:visible]
#     visible_hover = hover_texts[:visible]
#     visible_colors = colors[:visible]
#     visible_sizes = sizes[:visible]

#     edge_x, edge_y = [], []
#     for i in range(1, visible):
#         edge_x += [x[0], x[i], None]
#         edge_y += [y[0], y[i], None]

#     fig = go.Figure()

#     fig.add_trace(go.Scatter(
#         x=edge_x, y=edge_y,
#         mode='lines',
#         line=dict(width=1.5, color='#888'),
#         hoverinfo='none'
#     ))

#     fig.add_trace(go.Scatter(
#         x=visible_x, y=visible_y,
#         mode='markers+text',
#         text=visible_text,
#         textposition='bottom center',
#         hovertext=visible_hover,
#         hoverinfo='text',
#         marker=dict(
#             size=visible_sizes,
#             color=visible_colors,
#             line=dict(width=2, color='white')
#         ),
#         textfont=dict(size=12, color='black')
#     ))

#     fig.update_layout(
#         title=dict(text="🚀 Strategic Opportunity Map", font=dict(size=20, family='Arial Black')),
#         showlegend=False,
#         xaxis=dict(visible=False),
#         yaxis=dict(visible=False),
#         height=550,
#         plot_bgcolor='rgba(0,0,0,0)',
#         paper_bgcolor='rgba(0,0,0,0)',
#         margin=dict(l=20, r=20, t=80, b=20)
#     )
#     return fig

# # ==========================
# # 3D MARKET LANDSCAPE
# # ==========================
# def create_3d_market(result):
#     retrieved_docs = result.get("retrieved_docs", [])
#     idea = result["idea"]

#     if not retrieved_docs:
#         return go.Figure()

#     doc_texts = [doc["content"] for doc in retrieved_docs]
#     all_texts = [idea] + doc_texts
#     all_embeddings = embedding_model.encode(all_texts)

#     if SKLEARN_AVAILABLE:
#         pca = PCA(n_components=3)
#         coords = pca.fit_transform(all_embeddings)
#     else:
#         coords = random_projection(all_embeddings, 3)

#     df = pd.DataFrame({
#         'x': coords[:,0],
#         'y': coords[:,1],
#         'z': coords[:,2],
#         'label': ['Your Idea'] + [f"{doc['domain']} ({doc['year']})" for doc in retrieved_docs],
#         'size': [20] + [15]*len(retrieved_docs),
#         'color': ['red'] + ['blue']*len(retrieved_docs),
#         'hover': [f"<b>Your Idea</b><br>{idea}"] + 
#                  [f"<b>{doc['domain']} ({doc['year']})</b><br>{doc['content'][:100]}..." for doc in retrieved_docs]
#     })

#     fig = px.scatter_3d(df, x='x', y='y', z='z', text='label', size='size', color='color',
#                          hover_data={'hover': True, 'x': False, 'y': False, 'z': False, 'color': False, 'size': False},
#                          title="🧠 Market Landscape (3D Similarity Space)")
#     fig.update_traces(marker=dict(line=dict(width=2, color='white')),
#                       textposition='top center')
#     fig.update_layout(height=550, showlegend=False)
#     return fig

# # ==========================
# # KPI PANEL (2‑row layout)
# # ==========================
# def create_kpi_panel(df):
#     fig = go.Figure()
#     metrics = {
#         "Avg Growth": df["growth_rate"].mean(),
#         "Innovation Strength": df["innovation_score"].mean(),
#         "Market Potential": df["market_potential"].mean(),
#         "Strategic Advantage": (df["innovation_score"] + df["market_potential"] - df["competition_level"]).mean()
#     }

#     fig.add_trace(go.Indicator(
#         mode="number",
#         value=round(metrics["Avg Growth"], 2),
#         title={"text": "Avg Growth"},
#         domain={'x': [0, 0.45], 'y': [0.55, 1]},
#         number={"font": {"size": 36}}
#     ))
#     fig.add_trace(go.Indicator(
#         mode="number",
#         value=round(metrics["Innovation Strength"], 2),
#         title={"text": "Innovation Strength"},
#         domain={'x': [0.55, 1], 'y': [0.55, 1]},
#         number={"font": {"size": 36}}
#     ))
#     fig.add_trace(go.Indicator(
#         mode="number",
#         value=round(metrics["Market Potential"], 2),
#         title={"text": "Market Potential"},
#         domain={'x': [0, 0.45], 'y': [0, 0.45]},
#         number={"font": {"size": 36}}
#     ))
#     fig.add_trace(go.Indicator(
#         mode="number",
#         value=round(metrics["Strategic Advantage"], 2),
#         title={"text": "Strategic Advantage"},
#         domain={'x': [0.55, 1], 'y': [0, 0.45]},
#         number={"font": {"size": 36}}
#     ))

#     fig.update_layout(
#         template="plotly_dark",
#         height=320,
#         margin=dict(t=20, b=20, l=20, r=20)
#     )
#     return fig

# # ==========================
# # POSITIONING MATRIX
# # ==========================
# def create_positioning_matrix(df):
#     fig = px.scatter(
#         df,
#         x="competition_level",
#         y="innovation_score",
#         size="market_potential",
#         color="growth_rate",
#         hover_data=["domain","category","region","year"],
#         title="Strategic Positioning Matrix",
#         size_max=50,
#         color_continuous_scale="Turbo"
#     )
#     fig.update_traces(
#         marker=dict(line=dict(width=1, color='white')),
#         hovertemplate="<b>%{customdata[0]}</b><br>Category: %{customdata[1]}<br>Region: %{customdata[2]}<br>Year: %{customdata[3]}<br>Growth: %{color:.1f}<extra></extra>"
#     )
#     fig.update_layout(
#         template="plotly_dark",
#         height=400,
#         hovermode="closest",
#         transition={"duration": 500}
#     )
#     return fig

# # ==========================
# # GROWTH BY YEAR (YEAR SLIDER)
# # ==========================
# def create_growth_by_year(df, selected_year):
#     if "year" not in df.columns or df.empty:
#         fig = go.Figure()
#         fig.add_annotation(text="No data available", showarrow=False)
#         fig.update_layout(template="plotly_dark", height=400)
#         return fig

#     df_filtered = df[df["year"] == selected_year].copy()
#     if df_filtered.empty:
#         fig = go.Figure()
#         fig.add_annotation(text=f"No data for year {selected_year}", showarrow=False)
#         fig.update_layout(template="plotly_dark", height=400)
#         return fig

#     fig = px.bar(
#         df_filtered,
#         x="domain",
#         y="growth_rate",
#         color="growth_rate",
#         title=f"Domain Growth - {selected_year}",
#         color_continuous_scale="Viridis",
#         range_y=[0, 100],
#         hover_data=["category", "region"]
#     )
#     fig.update_traces(
#         hovertemplate="<b>%{x}</b><br>Growth: %{y:.1f}<br>Category: %{customdata[0]}<br>Region: %{customdata[1]}<extra></extra>"
#     )
#     fig.update_layout(
#         template="plotly_dark",
#         height=400,
#         xaxis_title="Domain",
#         yaxis_title="Growth Rate (%)"
#     )
#     return fig

# # ==========================
# # MARKET HEATMAP
# # ==========================
# def create_heatmap(df):
#     if "year" not in df.columns:
#         return go.Figure()
#     pivot = df.pivot_table(values="market_potential", index="domain", columns="year")
#     fig = px.imshow(
#         pivot,
#         color_continuous_scale="Magma",
#         title="Market Potential Heatmap",
#         aspect="auto",
#         labels=dict(x="Year", y="Domain", color="Potential")
#     )
#     fig.update_layout(template="plotly_dark", height=400, transition={"duration": 500})
#     fig.update_traces(hovertemplate="Year: %{x}<br>Domain: %{y}<br>Potential: %{z:.1f}<extra></extra>")
#     return fig

# # ==========================
# # OPPORTUNITY RADAR
# # ==========================
# def create_opportunity_radar(df):
#     if df.empty:
#         return go.Figure()
#     df["advantage"] = df["innovation_score"] + df["market_potential"] - df["competition_level"]
#     top = df.sort_values("advantage", ascending=False).iloc[0]
#     categories = ["Growth", "Innovation", "Market", "Competitive Advantage"]
#     values = [
#         top["growth_rate"],
#         top["innovation_score"],
#         top["market_potential"],
#         100 - top["competition_level"]
#     ]
#     fig = go.Figure()
#     fig.add_trace(go.Scatterpolar(
#         r=values,
#         theta=categories,
#         fill="toself",
#         marker=dict(color='rgba(102, 126, 234, 0.8)'),
#         hovertemplate="%{theta}: %{r:.1f}<extra></extra>"
#     ))
#     fig.update_layout(
#         template="plotly_dark",
#         polar=dict(radialaxis=dict(range=[0,100], visible=True)),
#         title=f"Top Opportunity: {top['domain']}",
#         height=400,
#         transition={"duration": 500}
#     )
#     return fig

# # ==========================
# # PDF GENERATION
# # ==========================
# def generate_pdf(result, radial_fig):
#     if not PDF_AVAILABLE:
#         return None

#     pdf = FPDF()
#     pdf.add_page()

#     if LOGO_PATH and os.path.exists(LOGO_PATH):
#         pdf.image(LOGO_PATH, x=10, y=8, w=30)
#     pdf.set_font("Arial", 'B', 16)
#     pdf.cell(0, 10, BRAND_NAME, ln=1, align='C')
#     pdf.ln(10)

#     pdf.set_font("Arial", 'B', 12)
#     pdf.cell(0, 10, "Executive Summary", ln=1)
#     pdf.set_font("Arial", '', 10)
#     summary = f"""
# Project Score: {result['project_score']}/10
# Feasibility: {result['feasibility']}
# Timeline: {result['estimated_timeline']}
# Market Similarity: {result['similarity_score']}

# Market Trend Summary:
# {result['market_trend_summary']}

# Uniqueness Analysis:
# {result['uniqueness_analysis']}

# Top Risks:
# {chr(10).join(['- ' + r for r in result['risks']])}

# Actionable Steps:
# {chr(10).join(['- ' + s for s in result['actionable_steps']])}
# """
#     pdf.multi_cell(0, 5, summary)
#     pdf.ln(5)

#     img_bytes = radial_fig.to_image(format="png")
#     with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_img:
#         tmp_img.write(img_bytes)
#         img_path = tmp_img.name
#     pdf.image(img_path, w=180)
#     os.unlink(img_path)
#     pdf.ln(5)

#     pdf.set_font("Arial", 'B', 12)
#     pdf.cell(0, 10, "Structured Report (JSON)", ln=1)
#     pdf.set_font("Courier", '', 8)
#     pdf.multi_cell(0, 4, json.dumps(result, indent=2))

#     with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_pdf:
#         pdf.output(tmp_pdf.name)
#         return tmp_pdf.name

# def prepare_pdf(result, radial_fig):
#     if not result or not radial_fig:
#         return None
#     return generate_pdf(result, radial_fig)

# # ==========================
# # UPDATE FUNCTIONS
# # ==========================
# def update_radial_and_state(result, step):
#     if not result:
#         return go.Figure(), go.Figure()
#     new_fig = create_radial_graph(result, step)
#     return new_fig, new_fig

# def update_live_score(result_state):
#     if result_state:
#         new_score = result_state.get("similarity_score", 0.5)
#         variation = np.random.uniform(-0.05, 0.05)
#         new_score = round(max(0, min(1, new_score + variation)), 3)
#         result_state["similarity_score"] = new_score
#         return f"Live Similarity: {new_score}"
#     return "Live Similarity: N/A"

# def increment_step(current_step):
#     return min(current_step + 1, 8)

# def update_growth_by_year(result_state, selected_year):
#     if not result_state:
#         return go.Figure()
#     df = pd.DataFrame(result_state.get("market_analysis", []))
#     for col in ["growth_rate","innovation_score","market_potential","competition_level"]:
#         if col in df.columns:
#             df[col] = pd.to_numeric(df[col], errors="coerce")
#     df.fillna(0, inplace=True)
#     return create_growth_by_year(df, selected_year)

# # ==========================
# # FULL PIPELINE
# # ==========================
# def full_pipeline(user_input):
#     try:
#         result = analyze_idea(user_input)

#         fig_radial = create_radial_graph(result, visible_nodes=1)
#         fig_3d = create_3d_market(result)
#         fig_gauge = create_gauge(result)

#         df = pd.DataFrame(result["market_analysis"])
#         for col in ["growth_rate","innovation_score","market_potential","competition_level"]:
#             if col in df.columns:
#                 df[col] = pd.to_numeric(df[col], errors="coerce")
#         df.fillna(0, inplace=True)

#         years = []
#         if "year" in df.columns:
#             years = sorted(df["year"].unique())
#             years = [str(y) for y in years]

#         if df.empty:
#             fig_kpi = go.Figure()
#             fig_matrix = go.Figure()
#             fig_growth = create_growth_by_year(df, None)
#             fig_heat = go.Figure()
#             fig_radar_opp = go.Figure()
#         else:
#             fig_kpi = create_kpi_panel(df)
#             fig_matrix = create_positioning_matrix(df)
#             initial_year = str(df["year"].iloc[0]) if "year" in df.columns else None
#             fig_growth = create_growth_by_year(df, initial_year)
#             fig_heat = create_heatmap(df)
#             fig_radar_opp = create_opportunity_radar(df)

#         summary_md = f"""
# ### 📊 Key Metrics
# - **Project Score**: **{result['project_score']}/10**
# - **Feasibility**: **{result['feasibility']}**
# - **Timeline**: **{result['estimated_timeline']}**
# - **Market Similarity**: **{result['similarity_score']}**

# ### 📈 Market Trend Summary
# {result['market_trend_summary']}

# ### 💡 Uniqueness Analysis
# {result['uniqueness_analysis']}

# ### ⚠️ Top Risks
# {chr(10).join(['- ' + r for r in result['risks']])}

# ### ✅ Actionable Steps
# {chr(10).join(['- ' + s for s in result['actionable_steps']])}
# """

#         confetti_html = ""
#         if result['project_score'] > 8:
#             confetti_html = """
#             <script src="https://cdn.jsdelivr.net/npm/canvas-confetti@1"></script>
#             <script>
#                 confetti({particleCount: 100, spread: 70, origin: { y: 0.6 }});
#             </script>
#             """

#         return (summary_md, json.dumps(result, indent=4),
#                 fig_radial, fig_3d, fig_gauge,
#                 fig_kpi, fig_matrix, fig_growth, fig_heat, fig_radar_opp,
#                 result, fig_radial, confetti_html, years)
#     except Exception as e:
#         error_msg = f"❌ Error: {str(e)}"
#         empty_fig = go.Figure()
#         return (error_msg, "{}", empty_fig, empty_fig, empty_fig,
#                 empty_fig, empty_fig, empty_fig, empty_fig, empty_fig,
#                 {}, empty_fig, "", [])

# # ==========================
# # CUSTOM CSS – CLEAN & PROFESSIONAL
# # ==========================
# custom_css = """
# @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

# * {
#     box-sizing: border-box;
# }

# .gradio-container {
#     font-family: 'Inter', sans-serif;
#     background-color: #f8fafc;
#     color: white;
#     min-height: 100vh;
#     padding: 2rem 1rem !important;
# }

# /* Header */
# .premium-header {
#     background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
#     padding: 2rem;
#     border-radius: 1.5rem;
#     color: white;
#     text-align: center;
#     margin-bottom: 2rem;
#     box-shadow: 0 20px 30px -10px rgba(0,0,0,0.2);
#     border: 1px solid rgba(255,255,255,0.1);
# }

# .premium-header h1 {
#     font-size: 2.5rem;
#     font-weight: 700;
#     margin-bottom: 0.5rem;
#     letter-spacing: -0.02em;
#     text-color:white;
# }

# .premium-header p {
#     font-size: 1rem;
#     opacity: 0.8;
#     font-weight: 400;
#     text-color:white;
# }

# /* Input area */
# .input-container {
#     background: white;
#     border-radius: 1.5rem;
#     padding: 1.5rem;
#     box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
#     border: 1px solid #e2e8f0;
# }

# /* Buttons */
# .btn-primary {
#     background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%);
#     color: white;
#     border: none;
#     padding: 0.75rem 2rem;
#     border-radius: 2rem;
#     font-weight: 600;
#     cursor: pointer;
#     transition: all 0.2s;
#     box-shadow: 0 4px 6px -1px rgb(37 99 235 / 0.3);
#     width: 100%;
#     font-size: 1rem;
# }

# .btn-primary:hover {
#     transform: translateY(-2px);
#     box-shadow: 0 10px 15px -3px rgb(37 99 235 / 0.4);
# }

# .btn-secondary {
#     background: white;
#     color: #1e293b;
#     border: 1px solid #cbd5e1;
#     padding: 0.5rem 1rem;
#     border-radius: 2rem;
#     font-weight: 500;
#     cursor: pointer;
#     transition: all 0.2s;
# }

# .btn-secondary:hover {
#     background: #f1f5f9;
#     border-color: #94a3b8;
# }

# /* Cards */
# .premium-card {
#     background: white;
#     border-radius: 1.5rem;
#     padding: 1.5rem;
#     box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
#     border: 1px solid #e2e8f0;
#     height: fit-content;
# }

# /* Plot containers */
# .plot-container {
#     background: white;
#     border-radius: 1.5rem;
#     padding: 1rem;
#     box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
#     border: 1px solid #e2e8f0;
#     margin-bottom: 1rem;
# }

# /* Sliders */
# .gr-slider {
#     background: white;
#     border-radius: 1rem;
#     padding: 0.5rem 1rem;
#     border: 1px solid #e2e8f0;
#     margin: 1rem 0;
# }

# /* Microphone button */
# .mic-button {
#     background: #2563eb;
#     color: white;
#     border: none;
#     border-radius: 50%;
#     width: 3rem;
#     height: 3rem;
#     font-size: 1.5rem;
#     cursor: pointer;
#     transition: all 0.2s;
#     box-shadow: 0 4px 6px -1px rgb(37 99 235 / 0.3);
# }

# .mic-button:hover {
#     transform: scale(1.1);
#     background: #1e40af;
# }

# /* Responsive */
# @media (max-width: 768px) {
#     .gradio-container {
#         padding: 1rem !important;
#     }
#     .premium-header h1 {
#         font-size: 1.8rem;
#     }
# }

# /* Fade-in animation */
# @keyframes fadeInUp {
#     from {
#         opacity: 0;
#         transform: translateY(10px);
#     }
#     to {
#         opacity: 1;
#         transform: translateY(0);
#     }
# }

# .fade-in {
#     animation: fadeInUp 0.5s ease-out;
# }

# /* Live similarity text */
# .live-sim {
#     background: #f1f5f9;
#     padding: 0.5rem 1rem;
#     border-radius: 2rem;
#     font-size: 0.9rem;
#     color: #1e293b;
#     border: 1px solid #cbd5e1;
#     display: inline-block;
# }
# """

# # ==========================
# # GRADIO INTERFACE – CLEAN LAYOUT
# # ==========================
# with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as app:
#     # Premium header
#     gr.HTML(f"""
#     <div class="premium-header">
#         <h1 style="color:white;"> {BRAND_NAME}</h1>
#         <p style="color:white;">AI-Powered Market Intelligence • Real-Time Analysis • Strategic Insights</p>
#     </div>
#     """)

#     # Input row
#     with gr.Row(elem_classes="fade-in"):
#         with gr.Column(scale=4, elem_classes="input-container"):
#             idea_input = gr.Textbox(
#                 label="",
#                 placeholder="Enter your startup idea... e.g., AI platform for personalized mental health coaching",
#                 lines=2,
#                 container=False
#             )
#         with gr.Column(scale=1, min_width=120):
#             with gr.Row():
#                 gr.HTML("""
#                 <button class="mic-button" onclick="startVoiceInput()">🎤</button>
#                 <script>
#                     function startVoiceInput() {
#                         var recognition = new webkitSpeechRecognition() || new SpeechRecognition();
#                         recognition.lang = 'en-US';
#                         recognition.interimResults = false;
#                         recognition.maxAlternatives = 1;
#                         recognition.onresult = function(event) {
#                             var text = event.results[0][0].transcript;
#                             var inputField = document.querySelector('textarea[placeholder*="Enter your startup idea"]');
#                             if (inputField) {
#                                 inputField.value = text;
#                                 var event = new Event('input', { bubbles: true });
#                                 inputField.dispatchEvent(event);
#                             }
#                         }
#                         recognition.start();
#                     }
#                 </script>
#                 """)
#                 analyze_btn = gr.Button("Analyze", elem_classes="btn-primary", size="lg")

#     # States
#     result_state = gr.State()
#     radial_fig_state = gr.State()
#     years_state = gr.State()
#     live_display = gr.HTML()  # Will show live similarity

#     # Summary and JSON in two columns
#     with gr.Row():
#         with gr.Column(scale=1, elem_classes="premium-card fade-in"):
#             summary_output = gr.Markdown()
#         with gr.Column(scale=1, elem_classes="premium-card fade-in"):
#             json_output = gr.Code(label="📄 Structured Report", language="json", lines=10)

#     # First row of plots (original)
#     with gr.Row():
#         with gr.Column(elem_classes="plot-container"):
#             radial_output = gr.Plot(label="🗺️ Strategic Opportunity Map")
#         with gr.Column(elem_classes="plot-container"):
#             market_output = gr.Plot(label="🧠 3D Market Landscape")
#         with gr.Column(elem_classes="plot-container"):
#             gauge_output = gr.Plot(label="🎯 Project Score")

#     # Second row (KPI and Matrix)
#     with gr.Row():
#         with gr.Column(elem_classes="plot-container"):
#             kpi_output = gr.Plot(label="📊 KPI Panel")
#         with gr.Column(elem_classes="plot-container"):
#             matrix_output = gr.Plot(label="🎯 Positioning Matrix")

#     # Growth with year slider
#     with gr.Row():
#         with gr.Column(elem_classes="plot-container"):
#             year_slider = gr.Slider(
#                 minimum=0, maximum=0, step=1, value=0,
#                 label="Select Year", visible=False,
#                 elem_classes="gr-slider"
#             )
#             growth_output = gr.Plot(label="📈 Growth by Year")

#     # Third row (Heatmap and Radar)
#     with gr.Row():
#         with gr.Column(elem_classes="plot-container"):
#             heat_output = gr.Plot(label="🔥 Market Heatmap")
#         with gr.Column(elem_classes="plot-container"):
#             radar_opp_output = gr.Plot(label="⭐ Opportunity Radar")

#     # Controls row
#     with gr.Row(elem_classes="premium-card fade-in"):
#         step_slider = gr.Slider(minimum=1, maximum=8, step=1, value=1, label="Reveal Steps", elem_classes="gr-slider")
#         reveal_btn = gr.Button("Reveal Next Step", elem_classes="btn-secondary")
#         live_btn = gr.Button("Refresh Live Similarity", elem_classes="btn-secondary")
#         pdf_btn = gr.Button("📥 Download PDF Report", elem_classes="btn-primary")
#     pdf_output = gr.File(label="Download PDF", visible=True)  # Now visible
#     confetti_html = gr.HTML()

#     # Live similarity display as HTML
#     live_display = gr.HTML("""
#     <div class="live-sim">Live Similarity: N/A</div>
#     """)

#     # Event handlers (same as before, but adjusted for live_display as HTML)
#     def update_after_analysis(summary, json, radial, market, gauge, 
#                               kpi, matrix, growth, heat, radar_opp, 
#                               result, radial_fig, confetti, years):
#         if years and len(years) > 0:
#             return (summary, json, radial, market, gauge,
#                     kpi, matrix, growth, heat, radar_opp,
#                     result, radial_fig, confetti,
#                     gr.update(minimum=0, maximum=len(years)-1, step=1, 
#                              value=0, visible=True, label=f"Year: {years[0]}"),
#                     years)
#         else:
#             return (summary, json, radial, market, gauge,
#                     kpi, matrix, growth, heat, radar_opp,
#                     result, radial_fig, confetti,
#                     gr.update(visible=False), years)

#     analyze_btn.click(
#         fn=full_pipeline,
#         inputs=idea_input,
#         outputs=[summary_output, json_output,
#                  radial_output, market_output, gauge_output,
#                  kpi_output, matrix_output, growth_output, heat_output, radar_opp_output,
#                  result_state, radial_fig_state, confetti_html, years_state]
#     ).then(
#         fn=update_after_analysis,
#         inputs=[summary_output, json_output,
#                 radial_output, market_output, gauge_output,
#                 kpi_output, matrix_output, growth_output, heat_output, radar_opp_output,
#                 result_state, radial_fig_state, confetti_html, years_state],
#         outputs=[summary_output, json_output,
#                  radial_output, market_output, gauge_output,
#                  kpi_output, matrix_output, growth_output, heat_output, radar_opp_output,
#                  result_state, radial_fig_state, confetti_html, year_slider, years_state]
#     ).then(
#         fn=lambda rs: f"<div class='live-sim'>Live Similarity: {rs.get('similarity_score', 'N/A')}</div>",
#         inputs=result_state,
#         outputs=live_display
#     )

#     def on_year_change(result_state, slider_val, years_state):
#         if years_state and slider_val < len(years_state):
#             selected_year = years_state[slider_val]
#             return update_growth_by_year(result_state, selected_year)
#         return go.Figure()

#     year_slider.change(
#         fn=on_year_change,
#         inputs=[result_state, year_slider, years_state],
#         outputs=growth_output
#     )

#     def update_slider_label(slider_val, years_state):
#         if years_state and slider_val < len(years_state):
#             return gr.update(label=f"Year: {years_state[slider_val]}")
#         return gr.update(label="Select Year")

#     year_slider.change(
#         fn=update_slider_label,
#         inputs=[year_slider, years_state],
#         outputs=year_slider
#     )

#     step_slider.change(
#         fn=update_radial_and_state,
#         inputs=[result_state, step_slider],
#         outputs=[radial_output, radial_fig_state]
#     )

#     reveal_btn.click(
#         fn=increment_step,
#         inputs=step_slider,
#         outputs=step_slider
#     ).then(
#         fn=update_radial_and_state,
#         inputs=[result_state, step_slider],
#         outputs=[radial_output, radial_fig_state]
#     )

#     live_btn.click(
#         fn=update_live_score,
#         inputs=result_state,
#         outputs=live_display
#     )

#     pdf_btn.click(
#         fn=prepare_pdf,
#         inputs=[result_state, radial_fig_state],
#         outputs=pdf_output
#     )

#     gr.HTML("""
#     <div style="text-align: center; margin-top: 3rem; padding: 1rem; color: #64748b; font-size: 0.9rem;">
#         <p>Powered by Groq LLama 3.3 • FAISS • Sentence‑Transformers</p>
#     </div>
#     """)

# if __name__ == "__main__":
#     app.launch(share=True)