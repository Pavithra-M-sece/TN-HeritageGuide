# app.py
import json
import gradio as gr
from pathlib import Path
from src.rag_pipeline import RAGPipeline
from src.config import GOLD_DIR

# ── Startup ──────────────────────────────────────────────────
print('🚀 Starting Smart Tamil Nadu Heritage Guide...')
rag = RAGPipeline()

# ── Load Gold Data ────────────────────────────────────────────
GOLD_DATA = {}
for gf in (GOLD_DIR / 'summaries').glob('*.json'):
    with open(gf, encoding='utf-8') as f:
        rec = json.load(f)
        GOLD_DATA[rec['site_id']] = rec

# ── Build Site Cards ──────────────────────────────────────────
SITE_CARDS = []
for site_id, gold in sorted(GOLD_DATA.items()):
    ai      = gold.get('ai_generated_content', {})
    tourism = gold.get('tourism_info', {})
    facts   = gold.get('key_facts', {})
    summary = ai.get('heritage_summary',
                     gold.get('original_summary', ''))
    if summary and len(summary) > 300:
        summary = summary[:300] + '...'

    SITE_CARDS.append({
        'id':            site_id,
        'name':          gold['site_name'],
        'tamil_name':    gold.get('tamil_name', ''),
        'location':      facts.get('location', ''),
        'type':          facts.get('type', ''),
        'dynasty':       facts.get('dynasty', ''),
        'period':        facts.get('period', ''),
        'deity':         facts.get('deity', ''),
        'summary':       summary or f"{gold['site_name']} — Tamil Nadu.",
        'visiting_hours':tourism.get('visiting_hours', 'N/A'),
        'entry_fee':     tourism.get('entry_fee', 'N/A'),
        'best_time':     tourism.get('best_time', 'N/A'),
        'dress_code':    tourism.get('dress_code', 'N/A'),
        'accessibility': tourism.get('accessibility', 'N/A'),
        'tip':           tourism.get('tip_for_visitors', ''),
        'suitable_for':  ', '.join(tourism.get('suitable_for', [])),
        'local_food':    ', '.join(tourism.get('local_food', [])),
        'tags':          facts.get('tags', []),
        'nearby':        ', '.join(
                             tourism.get('nearby_attractions', [])[:3]
                         ),
    })

SITE_NAMES = [c['name'] for c in SITE_CARDS]
print(f'✅ {len(SITE_CARDS)} heritage sites loaded')

# ── Site Card Renderer ────────────────────────────────────────
def render_site_card(card):
    tags_html = ' '.join([
        f'<span style="background:#8B4513;color:white;'
        f'padding:3px 10px;border-radius:20px;font-size:0.78em;'
        f'margin:2px;display:inline-block;">{t}</span>'
        for t in card['tags'][:5]
    ])
    tip_html = (
        f'<div style="background:#fff3cd;'
        f'border-left:4px solid #ffc107;padding:10px;'
        f'border-radius:4px;margin:10px 0;font-size:0.88em;'
        f'color:#856404;">💡 <em>{card["tip"]}</em></div>'
    ) if card['tip'] else ''

    return f"""
    <div style="border:1px solid #d4a843;border-radius:12px;
         padding:20px;
         background:linear-gradient(135deg,#fff9f0,#fffdf8);
         box-shadow:0 2px 8px rgba(212,168,67,0.15);margin:10px 0;">
        <div style="border-bottom:2px solid #d4a843;
             margin-bottom:12px;padding-bottom:10px;">
            <h3 style="color:#8B4513;margin:0;font-size:1.3em;">
                🏛️ {card['name']}
            </h3>
            <p style="color:#c0392b;margin:4px 0 0 0;">
                {card['tamil_name']}
            </p>
        </div>
        <div style="display:flex;flex-wrap:wrap;gap:10px;
             margin:10px 0;font-size:0.85em;color:#555;">
            <span>📍 {card['location']}</span>
            <span>🏛️ {card['type']}</span>
            <span>👑 {card['dynasty']}</span>
            <span>📅 {card['period']}</span>
        </div>
        <p style="color:#333;line-height:1.6;margin:10px 0;">
            {card['summary']}
        </p>
        <div style="display:grid;grid-template-columns:1fr 1fr;
             gap:10px;background:#f9f3e3;padding:14px;
             border-radius:8px;margin:12px 0;font-size:0.9em;">
            <div><strong>⏰</strong> {card['visiting_hours']}</div>
            <div><strong>🎟️</strong> {card['entry_fee']}</div>
            <div><strong>☀️</strong> {card['best_time']}</div>
            <div><strong>👔</strong> {card['dress_code']}</div>
            <div><strong>♿</strong> {card['accessibility']}</div>
            <div><strong>👥</strong> {card['suitable_for']}</div>
        </div>
        {tip_html}
        <div style="margin-top:12px;">{tags_html}</div>
    </div>"""

def get_site_card_html(site_name):
    for card in SITE_CARDS:
        if card['name'] == site_name:
            return render_site_card(card)
    return '<p>Site not found</p>'

# ── Welcome & Suggestions ─────────────────────────────────────
WELCOME = [{
    'role': 'assistant',
    'content': (
        '🙏 வணக்கம்! Welcome to the Smart Tamil Nadu Heritage Guide!\n\n'
        'I can help you with:\n'
        '• 🏛️ History and architecture of heritage sites\n'
        '• ⏰ Visiting hours, entry fees, directions\n'
        '• 👨‍👩‍👧 Family-friendly recommendations\n'
        '• 🗺️ Personalized heritage routes\n'
        '• 🌐 Questions in Tamil or English\n\n'
        'What would you like to know? 😊'
    )
}]

SUGGESTED_QUESTIONS = [
    'Tell me about Brihadeeshwara Temple',
    'Best sites for families with elderly?',
    'Entry fees for UNESCO sites?',
    'Plan a 2-day Chola temple circuit',
    'Meenakshi temple visiting hours?',
    'How to reach Shore Temple from Chennai?',
    'Wheelchair accessible heritage sites?',
    'Local food near Madurai temples?',
    'தஞ்சாவூர் கோயில் பற்றி சொல்லுங்கள்',
    'மதுரை மீனாட்சி கோவிலுக்கு நேரம்?',
]

# ── UI Handler Functions ───────────────────────────────────────
def chat_respond_ui(message, history):
    if not message or not message.strip():
        return history, '', '', ''
    try:
        old_history = []
        user_msg = None
        for item in (history or []):
            if isinstance(item, dict):
                role    = item.get('role', '')
                content = item.get('content', '')
                if role == 'user':
                    user_msg = content
                elif role == 'assistant' and user_msg:
                    old_history.append([user_msg, content])
                    user_msg = None

        result = rag.query(message, old_history)

        if result['citations']:
            badges = ' '.join([
                f'<span style="background:#e8f5e9;'
                f'border:1px solid #4caf50;border-radius:6px;'
                f'padding:3px 10px;font-size:0.82em;'
                f'color:#2e7d32;margin:2px;display:inline-block;">'
                f'📚 {c}</span>'
                for c in result['citations']
            ])
            cit_html = f'<strong>Sources:</strong> {badges}'
        else:
            cit_html = ''

        met_html = (
            f'<span style="background:#f0f4ff;border-radius:6px;'
            f'padding:4px 10px;font-size:0.82em;color:#3949ab;'
            f'margin:2px;display:inline-block;">'
            f'⚡ {result["total_time"]}s</span> '
            f'<span style="background:#f0f4ff;border-radius:6px;'
            f'padding:4px 10px;font-size:0.82em;color:#3949ab;'
            f'margin:2px;display:inline-block;">'
            f'🔍 {result["retrieval_time"]}s</span> '
            f'<span style="background:#f0f4ff;border-radius:6px;'
            f'padding:4px 10px;font-size:0.82em;color:#3949ab;'
            f'margin:2px;display:inline-block;">'
            f'🌐 {result["language"].upper()}</span>'
        )

        new_history = list(history or []) + [
            {'role': 'user',      'content': message},
            {'role': 'assistant', 'content': result['answer']},
        ]
        return new_history, '', cit_html, met_html

    except Exception as e:
        err = f'⚠️ Error: {str(e)[:150]}'
        new_history = list(history or []) + [
            {'role': 'user',      'content': message},
            {'role': 'assistant', 'content': err},
        ]
        return new_history, '', '', ''

def suggest_click(suggestion, history):
    return chat_respond_ui(suggestion, history or list(WELCOME))

def clear_chat_fn():
    return list(WELCOME), '', '', ''

def plan_route_ui(traveller, days, city, interests_list):
    interests_str = ', '.join(interests_list) \
                    if interests_list else 'general heritage'
    query = (
        f"Plan a {int(days)}-day Tamil Nadu heritage trip for "
        f"{traveller} starting from {city}. "
        f"Interests: {interests_str}. "
        f"Day-by-day itinerary with sites, timings, food, hotels."
    )
    try:
        result = rag.query(query, [])
        footer = (
            f"\n\n---\n📚 {', '.join(result['citations'])} "
            f"| ⚡ {result['total_time']}s"
        )
        return [{'role': 'assistant',
                 'content': result['answer'] + footer}]
    except Exception as e:
        return [{'role': 'assistant',
                 'content': f'⚠️ Error: {str(e)[:100]}'}]

# ── Build Gradio App ──────────────────────────────────────────
CSS = """
.gradio-container { max-width: 1200px !important; }
footer { display: none !important; }
"""

with gr.Blocks(title='Tamil Nadu Heritage Guide') as demo:

    gr.HTML("""
    <div style="text-align:center;padding:25px;
         background:linear-gradient(135deg,#8B4513,#D2691E);
         border-radius:16px;margin-bottom:20px;color:white;">
        <h1 style="margin:0;font-size:2.2em;">
            🏛️ Smart Tamil Nadu Heritage Guide
        </h1>
        <p style="margin:8px 0 4px 0;font-size:1.1em;opacity:0.9;">
            AI-Powered Heritage Tourism Assistant
        </p>
        <p style="margin:0;font-size:0.9em;opacity:0.75;">
            Tamil + English | 10 Heritage Sites | RAG Pipeline
        </p>
    </div>
    """)

    with gr.Tabs():

        # TAB 1 — CHAT
        with gr.Tab('💬 Heritage Chat'):
            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        label='Heritage Guide',
                        height=450,
                        value=list(WELCOME)
                    )
                    with gr.Row():
                        msg_input = gr.Textbox(
                            placeholder='Ask in Tamil or English...',
                            label='Your Question',
                            lines=2,
                            scale=5
                        )
                        with gr.Column(scale=1, min_width=100):
                            send_btn = gr.Button(
                                '🚀 Send',
                                variant='primary',
                                size='lg'
                            )
                            clear_btn = gr.Button(
                                '🗑️ Clear',
                                variant='secondary',
                                size='sm'
                            )
                    citations_out = gr.HTML(label='Sources')
                    metrics_out   = gr.HTML(label='Performance')

                with gr.Column(scale=1, min_width=200):
                    gr.Markdown('### 💡 Try These')
                    for s in SUGGESTED_QUESTIONS:
                        sb = gr.Button(s, size='sm', variant='secondary')
                        sb.click(
                            fn=suggest_click,
                            inputs=[gr.State(s), chatbot],
                            outputs=[chatbot, msg_input,
                                     citations_out, metrics_out]
                        )

            send_btn.click(
                fn=chat_respond_ui,
                inputs=[msg_input, chatbot],
                outputs=[chatbot, msg_input,
                         citations_out, metrics_out]
            )
            msg_input.submit(
                fn=chat_respond_ui,
                inputs=[msg_input, chatbot],
                outputs=[chatbot, msg_input,
                         citations_out, metrics_out]
            )
            clear_btn.click(
                fn=clear_chat_fn,
                outputs=[chatbot, msg_input,
                         citations_out, metrics_out]
            )

        # TAB 2 — SITE BROWSER
        with gr.Tab('🗺️ Explore Sites'):
            gr.Markdown(
                '### Browse all 10 Tamil Nadu Heritage Sites'
            )
            with gr.Row():
                site_dropdown = gr.Dropdown(
                    choices=SITE_NAMES,
                    value=SITE_NAMES[0],
                    label='Select Heritage Site',
                    scale=3
                )
                view_btn = gr.Button(
                    '🔍 View',
                    variant='primary',
                    scale=1
                )
            site_display = gr.HTML(
                value=get_site_card_html(SITE_NAMES[0])
            )
            site_dropdown.change(
                fn=get_site_card_html,
                inputs=[site_dropdown],
                outputs=[site_display]
            )
            view_btn.click(
                fn=get_site_card_html,
                inputs=[site_dropdown],
                outputs=[site_display]
            )
            gr.Markdown('#### ⚡ Quick Navigate:')
            with gr.Row():
                for card in SITE_CARDS[:5]:
                    qb = gr.Button(card['name'][:18], size='sm')
                    qb.click(
                        fn=lambda n=card['name']: (
                            n, get_site_card_html(n)
                        ),
                        outputs=[site_dropdown, site_display]
                    )
            with gr.Row():
                for card in SITE_CARDS[5:]:
                    qb = gr.Button(card['name'][:18], size='sm')
                    qb.click(
                        fn=lambda n=card['name']: (
                            n, get_site_card_html(n)
                        ),
                        outputs=[site_dropdown, site_display]
                    )

        # TAB 3 — ROUTE PLANNER
        with gr.Tab('🗓️ Route Planner'):
            gr.Markdown(
                '### 🎯 Personalized Heritage Route Planner'
            )
            with gr.Row():
                with gr.Column():
                    traveller_type = gr.Radio(
                        choices=[
                            'Family with elderly',
                            'Young couple',
                            'Solo traveller',
                            'School group',
                            'Photography enthusiast',
                            'Pilgrimage group'
                        ],
                        label='👥 Traveller Type',
                        value='Family with elderly'
                    )
                    num_days = gr.Slider(
                        minimum=1, maximum=5,
                        value=2, step=1,
                        label='📅 Number of Days'
                    )
                    start_city = gr.Dropdown(
                        choices=[
                            'Chennai', 'Madurai', 'Coimbatore',
                            'Trichy', 'Salem', 'Thanjavur'
                        ],
                        label='🚉 Starting City',
                        value='Chennai'
                    )
                    interests = gr.CheckboxGroup(
                        choices=[
                            'UNESCO Sites',
                            'Chola Architecture',
                            'Pallava Architecture',
                            'Pilgrimage',
                            'Coastal Sites',
                            'Photography',
                            'Local Food',
                            'Budget Travel'
                        ],
                        label='❤️ Interests',
                        value=['UNESCO Sites', 'Chola Architecture']
                    )
                    plan_btn = gr.Button(
                        '🗺️ Generate Route!',
                        variant='primary',
                        size='lg'
                    )
                with gr.Column():
                    route_output = gr.Chatbot(
                        label='Your Itinerary',
                        height=500,
                        value=[{
                            'role': 'assistant',
                            'content': (
                                '👆 Fill preferences '
                                'and click Generate!'
                            )
                        }]
                    )
            plan_btn.click(
                fn=plan_route_ui,
                inputs=[traveller_type, num_days,
                        start_city, interests],
                outputs=[route_output]
            )

        # TAB 4 — ABOUT
        with gr.Tab('ℹ️ About'):
            gr.Markdown("""
## 🏛️ Smart Tamil Nadu Heritage Guide
End-to-end **Data Engineering + GenAI** portfolio project.

**Tech Stack:** FAISS · BM25 · sentence-transformers
· CrossEncoder · Groq LLaMA 3.1 · Gradio

**Pipeline:** Bronze → Silver → Gold → RAG → Web App

*Built with ❤️ for Tamil Nadu cultural heritage*
            """)

if __name__ == '__main__':
    print('🚀 Launching Smart Tamil Nadu Heritage Guide...', flush=True)
    print('🌐 Open http://localhost:7860 in your browser to interact with the app.', flush=True)
    demo.launch(
        server_name='0.0.0.0',
        server_port=7860,
        share=False,
        css=CSS,
        theme=gr.themes.Soft()
    )