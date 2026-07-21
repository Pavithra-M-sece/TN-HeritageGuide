
# src/rag_pipeline.py
import time
from groq import Groq
from src.config import GROQ_API_KEY, GROQ_MODEL
from src.retriever import HybridRetriever

TAMIL_KEYWORDS = {
    'தஞ்சாவூர்':      'Thanjavur Brihadeeshwara Temple Chola',
    'பெரிய கோயில்':   'Brihadeeshwara Temple Thanjavur UNESCO',
    'மதுரை':           'Madurai Meenakshi Amman Temple',
    'மீனாட்சி':        'Meenakshi Amman Temple Madurai',
    'ராமேஸ்வரம்':     'Ramanathaswamy Temple Rameswaram',
    'சிதம்பரம்':       'Chidambaram Nataraja Temple',
    'வேலூர்':          'Vellore Fort Vijayanagara',
    'கஞ்சிபுரம்':     'Kanchipuram Kailasanathar Temple Pallava',
    'மகாபலிபுரம்':    'Shore Temple Mahabalipuram Pallava',
    'திருச்செங்கோடு': 'Tiruchengode Ardhanariswarar Temple',
    'கோயில்':          'temple heritage Tamil Nadu',
    'நேரம்':           'visiting hours timings',
    'கட்டணம்':         'entry fee ticket price',
    'வரலாறு':          'history heritage ancient dynasty',
}

SYSTEM_PROMPT = """You are an expert Tamil Nadu heritage tourism guide.
Help tourists understand and plan visits to Tamil Nadu heritage sites.
Always cite sources using [Site Name] format.
Be warm, enthusiastic and helpful.
End with one follow-up suggestion."""


class RAGPipeline:

    def __init__(self):
        self.retriever = HybridRetriever()
        self.groq      = Groq(api_key=GROQ_API_KEY)
        print('✅ RAG Pipeline ready')

    def enhance_tamil(self, query):
        enhanced = query
        for word, equiv in TAMIL_KEYWORDS.items():
            if word in query:
                enhanced = f"{query} {equiv}"
        return enhanced

    def detect_language(self, text):
        tamil_chars = sum(
            1 for c in text if '\u0B80' <= c <= '\u0BFF'
        )
        return 'tamil' if tamil_chars > 2 else 'english'

    def query(self, user_query, history=None):
        t0   = time.time()
        lang = self.detect_language(user_query)
        enhanced = self.enhance_tamil(user_query)

        retrieved      = self.retriever.retrieve(enhanced)
        retrieval_time = time.time() - t0

        context_parts = []
        for i, chunk in enumerate(retrieved, 1):
            context_parts.append(
                f"[SOURCE {i}: {chunk['site_name']} - "
                f"{chunk['chunk_type'].upper()}]\n{chunk['text']}"
            )
        context_str = '\n\n'.join(context_parts)

        history_str = ''
        if history:
            for h in history[-3:]:
                history_str += (
                    f"User: {h[0]}\n"
                    f"Assistant: {h[1][:150]}...\n"
                )

        lang_note = (
            '\nIMPORTANT: Reply in Tamil language.'
            if lang == 'tamil' else ''
        )

        prompt = (
            f"CONTEXT:\n{context_str}\n\n"
            f"{history_str}\n"
            f"TOURIST QUERY: {user_query}"
            f"{lang_note}\n\n"
            f"Answer with [Site Name] citations. Be helpful!"
        )

        t_gen = time.time()
        response = self.groq.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user',   'content': prompt}
            ],
            max_tokens=800,
            temperature=0.4
        )
        generation_time = time.time() - t_gen
        total_time      = time.time() - t0

        answer    = response.choices[0].message.content.strip()
        citations = list(set([c['site_name'] for c in retrieved]))

        return {
            'answer':          answer,
            'citations':       citations,
            'retrieved':       retrieved,
            'retrieval_time':  round(retrieval_time, 2),
            'generation_time': round(generation_time, 2),
            'total_time':      round(total_time, 2),
            'language':        lang,
        }
