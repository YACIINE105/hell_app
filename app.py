import streamlit as st
import google.generativeai as genai
import speech_recognition as sr
from gtts import gTTS
from audiorecorder import audiorecorder
import streamlit.components.v1 as components
import io
import re
import base64
import time

# --- الإعدادات الأولية ---
st.set_page_config(
    layout="centered",
    page_title="مساعد التاريخ المصري",
    page_icon="🏛️"
)

# تحميل مفتاح Gemini API من st.secrets
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
except KeyError:
    st.error("لم يتم العثور على مفتاح GEMINI_API_KEY. يرجى إضافته إلى .streamlit/secrets.toml")
    st.stop()
except Exception as e:
    st.error(f"حدث خطأ أثناء إعداد واجهة Gemini: {e}")
    st.stop()

# --- إعداد نموذج Gemini ---
generation_config = {
    "temperature": 0.7,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 2048,
}
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

system_instruction = """أنت مساعد ذكي متخصص في التاريخ المصري والشخصيات التاريخية المصرية.

قواعد مهمة جداً:
1. استخدم اللغة العربية الفصحى في جميع إجاباتك
2. اكتب إجابتك على شكل نقاط منفصلة، كل نقطة في سطر جديد
3. كل نقطة يجب أن تكون جملة كاملة ومفيدة (جملة أو جملتين)
4. اكتب من 3 إلى 5 نقاط فقط
5. لا تكتب ترحيب في البداية - ابدأ مباشرة بالمعلومات
6. ابدأ كل نقطة بـ "•" أو "-"

مثال على الرد المطلوب:
• توت عنخ آمون كان فرعوناً مصرياً حكم مصر وهو في التاسعة من عمره
• اكتشف هوارد كارتر مقبرته عام 1922 وكانت مليئة بالكنوز الثمينة
• تعتبر المقبرة من أهم الاكتشافات الأثرية في التاريخ
• توفي في سن التاسعة عشرة والسبب لا يزال غامضاً

تذكر: نقاط قصيرة ومفيدة باللغة العربية الفصحى!"""

model = genai.GenerativeModel(
    model_name="gemini-2.5-flash",
    generation_config=generation_config,
    system_instruction=system_instruction,
    safety_settings=safety_settings
)


# --- الدوال المساعدة ---

def transcribe_audio(audio_segment):
    """تحويل الصوت إلى نص عربي"""
    recognizer = sr.Recognizer()
    try:
        wav_bytes = audio_segment.export(format="wav").read()
        audio_data = io.BytesIO(wav_bytes)
        with sr.AudioFile(audio_data) as source:
            audio = recognizer.record(source)
        text = recognizer.recognize_google(audio, language="ar-SA")
        return text
    except sr.UnknownValueError:
        return "لم أستطع فهم الصوت. يرجى المحاولة مرة أخرى."
    except sr.RequestError as e:
        return f"خطأ في خدمة التعرف على الكلام: {e}"
    except Exception as e:
        return f"خطأ غير متوقع في معالجة الصوت: {e}"


def generate_tts_audio(text):
    """تحويل النص إلى صوت"""
    try:
        tts = gTTS(text=text, lang='ar', slow=False)
        audio_fp = io.BytesIO()
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)
        return audio_fp.read()
    except Exception as e:
        st.error(f"حدث خطأ أثناء إنشاء الصوت: {e}")
        return None


def extract_bullet_points(text):
    """استخراج النقاط من النص"""
    lines = text.split('\n')
    bullets = []

    for line in lines:
        line = line.strip()
        line = re.sub(r'^[•\-\*]\s*', '', line)
        if line and len(line) > 10:
            bullets.append(line)

    return bullets if bullets else [text]


def get_gemini_response(prompt_text):
    """إرسال الرسالة لـ Gemini مرة واحدة فقط"""
    try:
        chat_session = st.session_state.chat_session
        response = chat_session.send_message(prompt_text)
        return response.text
    except Exception as e:
        return f"حدث خطأ أثناء التواصل مع Gemini: {e}"


def apply_egyptian_theme():
    """تطبيق الثيم المصري الفرعوني"""
    st.markdown("""
<style>
    /* خلفية فرعونية */
    .stApp {
        background: linear-gradient(135deg, #c49b63 0%, #8b6f47 50%, #c49b63 100%);
        background-image: 
            repeating-linear-gradient(90deg, rgba(0,0,0,.03) 0px, transparent 1px, transparent 2px, rgba(0,0,0,.03) 3px),
            repeating-linear-gradient(0deg, rgba(0,0,0,.03) 0px, transparent 1px, transparent 2px, rgba(0,0,0,.03) 3px);
    }

    /* الحاوية الرئيسية */
    .main {
        background-color: rgba(255, 248, 240, 0.95);
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        border: 3px solid #8b6f47;
    }

    /* العنوان */
    h1 {
        color: #8b4513 !important;
        text-align: center !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        padding: 1rem;
        background: linear-gradient(90deg, transparent, rgba(139,69,19,0.1), transparent);
        border-radius: 10px;
    }

    /* رسائل الشات */
    .stChatMessage {
        background-color: rgba(255, 255, 255, 0.9) !important;
        border-radius: 15px !important;
        padding: 1rem !important;
        margin: 0.5rem 0 !important;
        border: 2px solid #d4a574 !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1) !important;
    }

    /* رسالة المستخدم */
    [data-testid="stChatMessageContent"] {
        background-color: transparent !important;
    }

    /* الأزرار */
    .stButton button {
        background: linear-gradient(135deg, #c49b63, #8b6f47) !important;
        color: white !important;
        border: 2px solid #8b4513 !important;
        border-radius: 10px !important;
        font-weight: bold !important;
        padding: 0.75rem 1.5rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2) !important;
    }

    .stButton button:hover {
        background: linear-gradient(135deg, #8b6f47, #c49b63) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 12px rgba(0,0,0,0.3) !important;
    }

    /* مشغل الصوت */
    audio {
        border-radius: 10px !important;
        background-color: rgba(255, 248, 240, 0.5) !important;
    }

    /* الرسائل التنبيهية */
    .stAlert {
        background-color: rgba(255, 248, 240, 0.95) !important;
        border: 2px solid #c49b63 !important;
        border-radius: 10px !important;
        color: #8b4513 !important;
    }

    /* Spinner */
    .stSpinner > div {
        border-top-color: #8b4513 !important;
    }

    /* فاصل */
    hr {
        border-color: #c49b63 !important;
        opacity: 0.3 !important;
    }

    /* Caption */
    .caption {
        color: #8b6f47 !important;
        text-align: center !important;
    }

    /* أيقونة المساعد - نمط هيروغليفي */
    [data-testid="chatAvatarIcon-assistant"] {
        background: linear-gradient(135deg, #c49b63, #8b6f47) !important;
        border: 2px solid #8b4513 !important;
    }

    /* أيقونة المستخدم */
    [data-testid="chatAvatarIcon-user"] {
        background: linear-gradient(135deg, #4a90e2, #2e5c8a) !important;
    }

    /* تنسيق حقل الإدخال */
    .stTextInput input {
        background-color: rgba(139, 111, 71, 0.1) !important;
        border: 2px solid #c49b63 !important;
        border-radius: 10px !important;
        color: #8b4513 !important;
        font-weight: 500 !important;
    }

    .stTextInput input:focus {
        border-color: #8b4513 !important;
        box-shadow: 0 0 0 2px rgba(139, 69, 19, 0.2) !important;
    }
</style>
""", unsafe_allow_html=True)


def create_sequential_audio_player(audio_list):
    """إنشاء مشغل صوتي يشغل التسجيلات بالتتابع تلقائياً"""
    # هذا الكود مطابق تماماً للكود في app.py
    if not audio_list:
        return

    # تحويل الصوت لـ base64
    audio_data_list = []
    for audio_bytes in audio_list[:10]:
        if audio_bytes:
            b64 = base64.b64encode(audio_bytes).decode()
            audio_data_list.append(b64)

    if not audio_data_list:
        return

    # إنشاء قائمة بصيغة JavaScript
    audio_sources = ',\n'.join([f'        "data:audio/mp3;base64,{src}"' for src in audio_data_list])

    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            body {{
                font-family: Arial, sans-serif;
                direction: rtl;
                background: linear-gradient(135deg, #fff8f0 0%, #f5e6d3 100%);
                margin: 0;
                padding: 0;
            }}
            #player-container {{
                padding: 20px;
                background: linear-gradient(135deg, rgba(196, 155, 99, 0.2), rgba(139, 111, 71, 0.2));
                border-radius: 15px;
                margin: 10px 0;
                border: 2px solid #c49b63;
                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            }}
            audio {{
                width: 100%;
                margin: 10px 0;
                border-radius: 10px;
            }}
            #status {{
                text-align: center;
                color: #8b4513;
                font-size: 14px;
                margin: 10px 0;
                font-weight: bold;
            }}
        </style>
    </head>
    <body>
        <div id="player-container">
            <audio id="audio-player" controls autoplay>
                متصفحك لا يدعم تشغيل الصوت
            </audio>
            <div id="status">جاري التحميل...</div>
        </div>

        <script>
            const audioSources = [
{audio_sources}
            ];

            let currentIndex = 0;
            const player = document.getElementById('audio-player');
            const status = document.getElementById('status');

            function playNext() {{
                if (currentIndex < audioSources.length) {{
                    status.textContent = 'جاري تشغيل الجزء ' + (currentIndex + 1) + ' من ' + audioSources.length;
                    player.src = audioSources[currentIndex];
                    player.load();

                    // محاولة التشغيل
                    const playPromise = player.play();
                    if (playPromise !== undefined) {{
                        playPromise.catch(error => {{
                            console.log('خطأ في التشغيل:', error);
                            // قد يمنع المتصفح التشغيل التلقائي، 
                            // لكن وجود "controls" يسمح للمستخدم بالبدء
                        }});
                    }}

                    currentIndex++;
                }} else {{
                    status.textContent = '✅ انتهى التشغيل';
                }}
            }}

            // عند انتهاء التسجيل الحالي
            player.addEventListener('ended', function() {{
                playNext();
            }});

            // عند حدوث خطأ
            player.addEventListener('error', function() {{
                console.log('خطأ في تحميل الصوت، الانتقال للتالي');
                playNext();
            }});

            // بدء التشغيل
            playNext();
        </script>
    </body>
    </html>
    """

    components.html(html_code, height=150, scrolling=False)


# --- واجهة التطبيق ---

# تطبيق الثيم المصري
apply_egyptian_theme()

st.markdown('<h1 style="text-align: center;">🏛️ مساعد Gemini الصوتي - التاريخ المصري</h1>', unsafe_allow_html=True)

st.markdown("""
<div style="text-align: center; padding: 1rem; background: linear-gradient(90deg, transparent, rgba(139,69,19,0.1), transparent); border-radius: 10px; margin-bottom: 1rem;">
    <p style="color: #8b4513; font-size: 1.1rem; margin: 0;">
        🔺 اسأل عن الشخصيات التاريخية المصرية، وسأجيب عليك باللغة العربية الفصحى! 🔺
    </p>
</div>
""", unsafe_allow_html=True)

# --- إعداد Session State ---
if "chat_session" not in st.session_state:
    st.session_state.chat_session = model.start_chat(history=[])

if "display_history" not in st.session_state:
    st.session_state.display_history = []

if "current_audio_list" not in st.session_state:
    st.session_state.current_audio_list = []

if "is_active_chat" not in st.session_state:
    st.session_state.is_active_chat = False

if "processing" not in st.session_state:
    st.session_state.processing = False

if "last_audio_len" not in st.session_state:
    st.session_state.last_audio_len = 0

if "last_text_input" not in st.session_state:
    st.session_state.last_text_input = ""

if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

if "query_source" not in st.session_state:
    st.session_state.query_source = None

# --- عرض سجل المحادثة ---
for message in st.session_state.display_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- معالجة الاستعلام المعلق وعرض الرد قبل حقل الإدخال ---
if st.session_state.pending_query:
    user_text = st.session_state.pending_query
    query_source = st.session_state.query_source

    # عرض رسالة المستخدم
    with st.chat_message("user"):
        st.markdown(user_text)

    # عرض حالة المعالجة
    with st.chat_message("assistant"):
        # إذا كان المصدر صوتي، نعرض رسالة معالجة الصوت أولاً
        if query_source == 'audio':
            with st.spinner("⏳ جاري معالجة الصوت..."):
                pass

        # عرض حالة التفكير
        with st.spinner("🤔 Gemini يفكر في الرد..."):
            full_response = get_gemini_response(user_text)

        # استخراج النقاط
        bullets = extract_bullet_points(full_response)

        if bullets:
            # توليد الصوت أولاً قبل عرض النص
            with st.spinner("🎵 جاري تحويل الردود إلى صوت..."):
                audio_list = []
                for bullet in bullets[:10]:
                    audio = generate_tts_audio(bullet)
                    if audio:
                        audio_list.append(audio)

                st.session_state.current_audio_list = audio_list

            # الآن نعرض النص بعد أن أصبح الصوت جاهزاً
            full_text = "\n\n".join([f"• {bullet}" for bullet in bullets])
            st.markdown(full_text)

            # إضافة للسجل
            st.session_state.display_history.append({
                "role": "user",
                "content": user_text
            })
            st.session_state.display_history.append({
                "role": "assistant",
                "content": full_text
            })

    # عرض مشغل الصوت بعد رسالة المساعد مباشرة
    if st.session_state.current_audio_list:
        st.markdown("### 🔊 استمع للرد:")
        create_sequential_audio_player(st.session_state.current_audio_list)

        if len(st.session_state.current_audio_list) >= 10:
            st.info("🎯 وصلنا لحد معلومات كافية (10 نقاط)! هل تريد السؤال عن موضوع آخر؟")

    # إنهاء المعالجة
    st.session_state.pending_query = None
    st.session_state.query_source = None
    st.session_state.processing = False

    # st.rerun() # --- !! تم حذف هذا السطر !! ---
    # هذا هو التعديل الرئيسي. بحذف هذا السطر،
    # نسمح لمشغل الصوت بالعمل قبل أي إعادة تحميل للصفحة.
    # ستستمر الصفحة الآن بشكل طبيعي لعرض أدوات الإدخال أدناه.

# --- واجهة الإدخال ---
st.markdown("---")

# إنشاء الواجهة
if not st.session_state.processing:
    # صف واحد يحتوي على: زر التسجيل + حقل الإدخال
    input_col1, input_col2 = st.columns([1, 8])

    with input_col1:
        audio_bytes = audiorecorder("🎤", "⏺️")

    with input_col2:
        text_input = st.text_input(
            "اكتب سؤالك هنا أو اضغط على المايكروفون...",
            key="text_input",
            label_visibility="collapsed"
        )

    # أزرار التحكم
    btn_col1, btn_col2 = st.columns(2)

    with btn_col1:
        new_topic_btn = st.button("🔄 موضوع جديد", use_container_width=True, key="new_topic_main")

    with btn_col2:
        clear_chat_btn = st.button("🗑️ مسح المحادثة", use_container_width=True, key="clear_chat_main")

    # التحقق من أن التسجيل جديد وليس نفس التسجيل السابق
    current_audio_len = len(audio_bytes) if audio_bytes else 0
    is_new_recording = current_audio_len > 0 and current_audio_len != st.session_state.last_audio_len

    # التحقق من أن النص جديد وليس نفس النص السابق
    is_new_text = text_input and text_input != st.session_state.last_text_input

    # معالجة التسجيل الصوتي
    if audio_bytes and is_new_recording and not st.session_state.processing:
        st.session_state.last_audio_len = current_audio_len
        st.session_state.processing = True

        user_text = transcribe_audio(audio_bytes)

        if "خطأ" in user_text or "لم أستطع" in user_text:
            st.error(user_text)
            st.session_state.processing = False
        else:
            st.session_state.is_active_chat = True
            st.session_state.current_audio_list = []
            st.session_state.pending_query = user_text
            st.session_state.query_source = 'audio'
            st.rerun()

    # معالجة إدخال النص
    if is_new_text and not st.session_state.processing:
        st.session_state.processing = True
        st.session_state.last_text_input = text_input
        st.session_state.is_active_chat = True
        st.session_state.current_audio_list = []
        st.session_state.pending_query = text_input
        st.session_state.query_source = 'text'
        st.rerun()

    # معالجة الأزرار
    if new_topic_btn:
        st.session_state.current_audio_list = []
        st.session_state.processing = False
        st.success("تمام! اسأل سؤالك الجديد 🎤")

    if clear_chat_btn:
        st.session_state.chat_session = model.start_chat(history=[])
        st.session_state.display_history = []
        st.session_state.current_audio_list = []
        st.session_state.is_active_chat = False
        st.session_state.processing = False
        st.session_state.pending_query = None
        st.session_state.query_source = None
        st.session_state.last_audio_len = 0
        st.session_state.last_text_input = ""
        st.rerun()

else:
    # هذا سيعرض رسالة "جاري المعالجة" بينما يتم تنفيذ
    # الجزء الخاص بـ st.session_state.pending_query
    st.info("⏳ جاري المعالجة، انتظر من فضلك...")
