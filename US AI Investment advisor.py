
# import streamlit as st
# from langgraph.graph import StateGraph, END
# from langchain_openai import ChatOpenAI
# from dotenv import load_dotenv
# import os
# from typing import Literal, TypedDict

# # ========== CONFIG ==========
# load_dotenv()
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# openai_llm = ChatOpenAI(model_name="gpt-4o", temperature=0.2, api_key=OPENAI_API_KEY)

#code above is to run locally , it has been replaced with few other lines to work on streamlit

import streamlit as st
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from typing import Literal, TypedDict

# ========== CONFIG ==========
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
openai_llm = ChatOpenAI(model_name="gpt-4o", temperature=0.2, api_key=OPENAI_API_KEY)


# ========== STATE STRUCTURE ==========
class InvestorState(TypedDict):
    age: int
    income: float
    net_worth: float
    profession: str
    horizon: str
    anticipated_retirement_age: int
    risk: str
    goal: str
    scope: Literal["basic", "comprehensive"]
    marital_status: str
    children: int
    profile: str
    research_plan: str
    market_data: str
    macro_analysis: str
    portfolio: str
    proposal: str

# ========== NODES ==========
def analyze_investor_profile(state: InvestorState) -> InvestorState:
    prompt = f"""
    Analyze this investor profile:
    - Age: {state['age']}
    - Income: ₹{state['income']}
    - Net Worth: ₹{state['net_worth']}
    - Profession: {state['profession']}
    - Marital Status: {state['marital_status']}, Children: {state['children']}
    - Investment Horizon: {state['horizon']} (Anticipated retirement age: {state['anticipated_retirement_age']})
    - Risk Tolerance: {state['risk']}
    - Goal: {state['goal']}
    """
    result = openai_llm.invoke(prompt)
    state["profile"] = result.content
    return state

def plan_research(state: InvestorState) -> InvestorState:
    prompt = f"""Based on this detailed profile:
    {state['profile']}
    Suggest focused investment research areas..."""
    result = openai_llm.invoke(prompt)
    state["research_plan"] = result.content
    return state

def route_based_on_scope(state: InvestorState) -> list[str]:
    return ["fetch_market_data", "analyze_macro"] if state["scope"] == "comprehensive" else ["build_portfolio"]

def fetch_market_data(state: InvestorState) -> dict:
    result = openai_llm.invoke("Fetch current market data relevant for retirement and wealth-building.")
    return {"market_data": result.content}

def analyze_macro(state: InvestorState) -> dict:
    result = openai_llm.invoke("Analyze current macroeconomic trends impacting retirement.")
    return {"macro_analysis": result.content}

def build_portfolio(state: InvestorState) -> InvestorState:
    prompt = f"""Given:
    Profile: {state['profile']}
    Market Data: {state['market_data']}
    Macro Analysis: {state['macro_analysis']}
    Recommend portfolio allocation."""
    result = openai_llm.invoke(prompt)
    state["portfolio"] = result.content
    return state

def generate_proposal(state: InvestorState) -> InvestorState:
    prompt = f"""Proposal:
    Profile: {state['profile']}
    Research Plan: {state['research_plan']}
    Market Data: {state['market_data']}
    Macro Analysis: {state['macro_analysis']}
    Portfolio: {state['portfolio']}"""
    result = openai_llm.invoke(prompt)
    state["proposal"] = result.content
    return state

# ========== GRAPH BUILDING ==========
graph = StateGraph(InvestorState)
graph.add_node("analyze_profile", analyze_investor_profile)
graph.add_node("plan_research", plan_research)
graph.add_node("fetch_market_data", fetch_market_data)
graph.add_node("analyze_macro", analyze_macro)
graph.add_node("build_portfolio", build_portfolio)
graph.add_node("generate_proposal", generate_proposal)

graph.set_entry_point("analyze_profile")
graph.add_edge("analyze_profile", "plan_research")
graph.add_conditional_edges("plan_research", route_based_on_scope, ["fetch_market_data", "analyze_macro", "build_portfolio"])
graph.add_edge(["fetch_market_data", "analyze_macro"], "build_portfolio")
graph.add_edge("build_portfolio", "generate_proposal")
graph.add_edge("generate_proposal", END)
compiled_graph = graph.compile()

# ========== STREAMLIT APP ==========
st.set_page_config(page_title="Rizq Advisor", layout="wide")
st.title("📊 Rizq Advisor - Personalized Investment Planner")

with st.form("user_input"):
    st.subheader("Enter Investor Details")
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100)
        income = st.number_input("Monthly Income (₹)", step=1000.0)
        net_worth = st.number_input("Net Worth (₹)", step=1000.0)
    with col2:
        profession = st.text_input("Profession")
        marital_status = st.selectbox("Marital Status", ["Single", "Married"])
        children = st.number_input("Number of Children", step=1)
    with col3:
        horizon = st.text_input("Investment Horizon", value="long-term")
        retirement_age = st.number_input("Anticipated Retirement Age", value=60)
        risk = st.selectbox("Risk Tolerance", ["low", "moderate", "high"])
    goal = st.text_input("Investment Goal", value="retirement")
    scope = st.selectbox("Analysis Scope", ["basic", "comprehensive"])
    submitted = st.form_submit_button("Generate Investment Plan")

if submitted:
    with st.spinner("Generating your personalized investment plan..."):
        init_input = {
            "age": age,
            "income": income,
            "net_worth": net_worth,
            "profession": profession,
            "marital_status": marital_status,
            "children": children,
            "horizon": horizon,
            "anticipated_retirement_age": retirement_age,
            "risk": risk,
            "goal": goal,
            "scope": scope,
            "profile": "",
            "research_plan": "",
            "market_data": "",
            "macro_analysis": "",
            "portfolio": "",
            "proposal": ""
        }
        result = compiled_graph.invoke(init_input)

        st.success("✅ Investment Proposal Generated!")

        st.subheader("1. Investor Profile")
        st.markdown(result["profile"])

        st.subheader("2. Research Plan")
        st.markdown(result["research_plan"])

        if scope == "comprehensive":
            st.subheader("3. Market Data")
            st.markdown(result["market_data"])

            st.subheader("4. Macro Analysis")
            st.markdown(result["macro_analysis"])

        st.subheader("5. Portfolio Allocation")
        st.markdown(result["portfolio"])

        st.subheader("6. Final Proposal")
        st.markdown(result["proposal"])
