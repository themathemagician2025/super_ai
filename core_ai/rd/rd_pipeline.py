# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

from .data_intake.intake_orchestrator import run_data_intake
from .llm_agents.summarizer import summarize_text
from .storage.vector_db import store_embeddings
from .dashboard.dashboard import show_dashboard
import logging

def run_rd_pipeline():
    logging.info('--- Starting R&D Pipeline ---')
    raw_data = run_data_intake()
    summaries = [summarize_text(d['content']) for d in raw_data]
    # TODO: Generate embeddings and store in vector DB
    # store_embeddings(summaries, embeddings)
    # TODO: Optionally update knowledge graph
    logging.info('--- R&D Pipeline Complete ---')

if __name__ == '__main__':
    run_rd_pipeline()
    # To launch dashboard, run: streamlit run super_ai/core_ai/rd/dashboard/dashboard.py 