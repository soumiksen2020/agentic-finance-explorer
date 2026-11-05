import json
import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, LLM
from crewai_tools import SerperDevTool
from tools import stock_price_analyzer
from pydantic import BaseModel, Field
from typing import List

import crewai.llms.cache as crew_cache

# Disable unsupported cache breakpoint injection for Groq
crew_cache.mark_cache_breakpoint = lambda msg: msg

class FinancialAnalysisOutput(BaseModel):
    ticker: str = Field(
        description="The stock ticker symbol analyzed"
    )
    technical_signal: str = Field(
        description="Overall technical trend: Bullish, Bearish, or Neutral"
    )
    sentiment_score: float = Field(
        description="Market sentiment score from 1.0 (Extreme Fear/Bearish) to 10.0 (Extreme Greed/Bullish)"
    )
    key_catalysts: List[str] = Field(
        description="Exactly 3 bullet points highlighting upcoming events, positive news, or fundamental strengths driving the stock."
    )
    risk_summary: List[str] = Field(
        description="Exactly 3 bullet points highlighting critical risks, bearish technicals, or fundamental weaknesses."
    )


load_dotenv()


def run_financial_analysis(ticker: str):

    ticker = ticker.strip().upper()

    if not ticker:
        raise ValueError("A valid stock ticker is required.")

    # 1. Tools & LLM Setup

    search_tool = SerperDevTool()

    raw_news_results = search_tool.run(
        search_query=f"{ticker} latest financial news November 2026"
    )

    # Parse the Serper response
    if isinstance(raw_news_results, str):
        try:
            parsed_news = json.loads(raw_news_results)
        except json.JSONDecodeError:
            parsed_news = {}
    else:
        parsed_news = raw_news_results

    # Retain only the four most relevant results
    organic_results = parsed_news.get("organic", [])[:4]

    news_search_results = "\n".join(
        f"""
        Headline: {item.get('title', 'N/A')}
        Summary: {item.get('snippet', 'N/A')[:220]}
        URL: {item.get('link', 'N/A')}
        """
        for item in organic_results
    )

    print("\n--- SERPER NEWS RESULTS FETCHED ---")
    print(news_search_results)

    llm = LLM(
        model="groq/openai/gpt-oss-120b",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0
    )

    # 2. Define Agents

    # Agent 1: Technical Analysis
    quant_analyst = Agent(
        role="Senior Quant Researcher",
        goal=f"Perform accurate technical analysis of {ticker}.",
        backstory="""You are a precise technical analyst at a Mumbai-based
        financial research firm. You specialize in stock price movements,
        RSI, moving averages, and technical market signals.""",
        tools=[stock_price_analyzer],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=3,
        max_retry_limit=1
    )

    # Agent 2: Financial News Research
    news_analyst = Agent(
        role="Financial News Correspondent",
        goal=f"Identify important financial news and market catalysts for {ticker}.",
        backstory=(
            "You are an experienced financial news researcher. "
            "Search results have already been retrieved using Serper "
            "and are supplied directly in your task description. "
            "Use only these supplied headlines, snippets and URLs. "
            "Do not attempt additional searches, browse webpages, "
            "open URLs or use file-opening tools. "
            "If the supplied evidence is insufficient, clearly state "
            "that the information could not be verified."
       ),
        tools=[],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=3,
        max_retry_limit=1
    )

    # Agent 3: Risk Assessment
    risk_manager = Agent(
        role="Chief Risk Officer (CRO)",
        goal=f"Identify significant investment risks and downside factors for {ticker}.",
        backstory="""You are a veteran risk manager at a leading Indian bank.

        You critically examine technical indicators, financial news,
        business developments, regulatory exposure, promoter-related
        concerns, and macroeconomic conditions.

        Your responsibility is to identify risks that may be overlooked
        by optimistic analysts.""",
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=3,
        max_retry_limit=1
    )

    # Agent 4: Fact Checking
    fact_checker = Agent(
        role="Senior Financial Fact-Checking Analyst",
        goal=f"Independently verify important financial claims and findings about {ticker}.",
        backstory="""You are a meticulous financial fact-checking specialist.

        You examine research findings, numerical claims, technical indicators,
        corporate announcements, and news reports.

        You carefully cross-check important claims against the
        technical data and news evidence supplied to you.

        You distinguish between:
        - Verified information supported by evidence.
        - Conflicting information from different sources.
        - Outdated information.
        - Claims that cannot be independently verified.

        Never invent evidence, sources, URLs, dates, or financial figures.
        Do not describe a claim as verified unless supporting evidence exists.
        If the supplied evidence is insufficient, explicitly mark the claim
        as unverified.""",
        tools=[],
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=2,
        max_retry_limit=1
       )

    # Agent 5: Final Report Generation
    report_writer = Agent(
        role="Senior Financial Research Report Writer",
        goal=f"Produce a clear, balanced, evidence-based financial analysis report for {ticker}.",
        backstory="""You are a senior financial research writer.

        You consolidate technical analysis, financial news research,
        risk assessment, and fact-checking findings into a concise
        professional research report.

        You prioritize verified information, distinguish facts from
        assumptions, and avoid unsupported investment conclusions.

        You must preserve the requested structured output format.
        Do not introduce claims that are not supported by the
        preceding research and verification.""",
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=3,
        max_retry_limit=1
    )

    # 3. Define Tasks

    # Task 1: Technical Analysis
    tech_task = Task(
        description=f"""
        Perform technical analysis for the stock {ticker}.

        Use the stock price analyzer to retrieve:
        - Latest available stock price.
        - RSI (14).
        - 20-day Simple Moving Average (MA20).

        Interpret the indicators and identify the overall technical trend.

        Clearly state the retrieved figures and avoid inventing missing data.
        """,
        expected_output="""
        Technical analysis containing the latest price, RSI, MA20,
        and an overall Bullish, Bearish, or Neutral technical signal.
        """,
        agent=quant_analyst
    )

    # Task 2: News Research
    news_task = Task(
        description=f"""
        Research financial news and corporate developments concerning {ticker}.

        Focus on news published during the last 7 days.

        Identify:
        - Important positive developments and catalysts.
        - Negative developments and potential concerns.
        - Earnings, regulatory, corporate, and industry-related events.

        Include source names, publication dates, and URLs wherever available.
        Distinguish recent developments from older background information.
        Do not fabricate news or sources.

        Search results retrieved directly from Serper:

        {news_search_results}

        Use only these supplied search results to identify recent
        financial developments. Do not attempt to call any search,
        browsing, or file-opening tool.

        If the supplied results do not contain sufficient evidence,
        explicitly state that the information could not be verified.
        """,
        expected_output="""
        A concise research summary of important financial news,
        including positive and negative developments, source names,
        dates, URLs where available, and an indicative sentiment assessment.
        """,
        agent=news_analyst,
        context=[tech_task]
    )

    # Task 3: Risk Assessment
    risk_task = Task(
        description=f"""
        Conduct a detailed risk assessment for {ticker}.

        Use the technical analysis and financial news research.

        Examine:
        - Technical weaknesses and downside signals.
        - Corporate and financial risks.
        - Regulatory and promoter-related concerns.
        - Industry and macroeconomic threats.

        Identify the three most important risks.

        Provide an indicative sentiment score between 1.0 and 10.0:
        1.0 = Extreme Fear / Bearish
        10.0 = Extreme Greed / Bullish

        Explain the reasoning behind the score.
        Do not invent financial facts or unsupported allegations.
        Base the assessment primarily on the supplied research.
        Do not assume that a risk exists merely because it is
        common in the financial industry.

        If the supplied evidence does not establish a specific
        corporate, regulatory, or promoter-related risk, identify
        it as a general potential risk rather than a confirmed issue.

        Do not repeat technical or news findings unnecessarily.
        """,
        expected_output="""
        A concise risk assessment containing:

        1. Exactly three important risks.
           Each risk must be explained in 1-2 sentences.
        2. One sentiment score between 1.0 and 10.0.
        3. A brief rationale for the sentiment score.

        Clearly distinguish evidence-supported risks from general
        industry risks and analytical assumptions.

        Do not introduce unsupported financial figures,
        macroeconomic forecasts, or speculative claims.
        """,
        agent=risk_manager,
        context=[tech_task, news_task]
    )


    # Task 4: Independent Fact Checking
    fact_check_task = Task(
        description=f"""
            Fact-check the financial analysis of {ticker} using the
            technical analysis and financial news research supplied
            by the preceding agents.

            Review:
            1. Technical Analysis Agent findings.
            2. Financial News Research Agent findings.

            Check the following:

            - Whether stock price, RSI and MA20 figures are consistent
              with the technical analysis output.
            - Whether news headlines, dates, figures and source names
              are accurately represented.
            - Whether the news findings support the claims made.
            - Identify conflicting numerical values or dates.
            - Flag unsupported statements and unverified allegations.
            - Identify outdated information.

            Classify each important claim as:
            VERIFIED, CONFLICTING, OUTDATED, or UNVERIFIED.

            Verification rules:

            - Use only the evidence supplied in the preceding task outputs.
            - Do not call external tools or attempt web searches.
            - Do not invent sources, URLs, dates or financial figures.
            - A claim is VERIFIED only when the supplied evidence
              directly supports it.
            - If a source snippet is incomplete or insufficient,
              classify the claim as UNVERIFIED.
            - Distinguish source-reported facts from analytical opinions.
            - Do not independently claim that a source is authoritative
              when its contents have not been examined.

            - Treat the September 2026 quarter and Q2 FY27 as the
              same reporting period. Do not describe them as separate
              quarters or consecutive earnings events.

            - If two sources report different profit growth percentages
              or profit figures for the same reporting period, flag
              the difference as CONFLICTING or UNVERIFIED unless
              the supplied evidence clearly explains the difference
              in accounting measure or reporting basis.

            - A claim appearing in a search-result snippet is not
              independently verified merely because an agent repeats it.

            - Do not mark an undated or incomplete Perplexity snippet
              as VERIFIED. Classify its financial claims as UNVERIFIED
              unless the supplied evidence establishes them.

            - Compare the Yahoo Finance price snapshot with the
              Quant Analyst's price. If the values differ, explicitly
              flag the discrepancy instead of claiming they match.

            - Do not treat two descriptions of the same event,
              such as an earnings announcement and the resulting
              share-price rally, as two independent earnings catalysts.

            - Never infer that an earnings result caused a share-price
              movement unless the supplied evidence directly establishes
              that relationship.

            - Do not create new financial claims.
            """,
      expected_output="""
            A concise fact-checking report containing:
            1. Verified, conflicting, outdated, and unverified claims.
            2. Technical figures checked.
            3. News claims and source discrepancies.
            4. Unsupported statements and unverified allegations.
            5. A clear distinction between facts and opinions.
      """,
      agent=fact_checker,
      context=[tech_task]
     )


    # Task 5: Final Financial Report
    report_task = Task(
        description=f"""
        Prepare the final financial analysis report for {ticker}.

        Consolidate findings from the available preceding task outputs:
        - Technical Analysis.
        - Risk Assessment.
        - Independent Fact Checking.

        Use the Fact Checking report for news-related claims.
        The original News Research output is not directly included
        in your context.

        - If the Fact Checker does not produce a completed report,
          do not label any claim as VERIFIED.
        - Treat CRO risk assessments as analytical opinions,
          not independently verified facts.

        Apply the following rules:

        1. Prioritize claims supported by the fact-checking results.
        2. Exclude fabricated, unsupported, or misleading claims.
        3. Clearly qualify conflicting or unverified information.
        4. Preserve important technical figures and their interpretation.
        5. Select exactly three important positive or neutral catalysts
           for key_catalysts.
        6. Select exactly three important risks for risk_summary.
        7. Do not present the analysis as guaranteed investment advice.

        Populate the final output fields:

        - ticker: {ticker}
        - technical_signal: Bullish, Bearish, or Neutral.
        - sentiment_score: A score between 1.0 and 10.0.
        - key_catalysts: Exactly three concise findings.
        - risk_summary: Exactly three concise findings.

        Where useful, include source names and verification qualifications
        inside the relevant catalyst or risk descriptions.

        Return only the requested structured output.
        """,
        expected_output="""
        A validated financial analysis JSON object containing ticker,
        technical_signal, sentiment_score, exactly three key_catalysts,
        and exactly three risk_summary entries.
        """,
        agent=report_writer,
        context=[tech_task, risk_task, fact_check_task],
        # output_json=FinancialAnalysisOutput
    )

    # 4. Assemble & Execute Crew

    financial_crew = Crew(
        agents=[
            quant_analyst,
            news_analyst,
            risk_manager,
            fact_checker,
            report_writer
        ],
        tasks=[
            tech_task,
            news_task,
            risk_task,
            fact_check_task,
            report_task
        ],
        verbose=True,
        process="sequential",
        max_rpm=4
    )

    return financial_crew.kickoff(inputs={"ticker": ticker})
