import re
from openai import OpenAI, BadRequestError

CACHE = {}


client = OpenAI()

PROMPT = "You are an expert translator.\nTranslate from {source_language} to {target_language}.\nProvide only the translation without explanations."



LANGUAGES = {
    "Arabic": "ar",
    "English": "en",
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Japanese": "ja",
    "Korean": "ko",
    "Thai": "th",
    "Romanian": 'ro',
    "Turkish": "tr",
    "Spanish": "es",
    "Simplified Chinese": "zh",
}

def translate_text(source_text, prompt, source_language, target_language, system_name):
    
  
    prompt = prompt.format(source_language=source_language, target_language=target_language).replace("ChineseSimplified", 'Simplified Chinese')
    
    messages=[
        {"role": "system", "content": prompt},
        {"role": "user", "content": source_text}
    ]
    
    response = client.chat.completions.create(
        model=system_name,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        messages=messages
    )

    response.choices[0].message.content = re.sub(
         r"\s([?.!])", r"\1", response.choices[0].message.content
         )
   
    return response.choices[0].message.content.strip()

global HAS_PRINTED 
HAS_PRINTED = False

def translate_cachable(t, p, from_lang, to_lang, s):
    key = (t, p, from_lang, to_lang, s)
    
    global HAS_PRINTED 
    
   
    if key not in CACHE:
        tries = 0
        while key not in CACHE:
            try:
                CACHE[key] = translate_text(t, p, from_lang, to_lang, s)
            except BadRequestError:
                print("error, retrying...")
                tries += 1
                if tries >= 20:
                    assert False
    else:
        if not HAS_PRINTED:
            print("Using cached translation...")
            HAS_PRINTED = True
        
    return CACHE[key]

def translate_gpt(in_text, in_lang, out_lang, sys_name='gpt-4o'):
   
    
    longer_languages = {'en': 'English', 'fr': 'French', 'de': 'German', 
                        'it': 'Italian', 'es': 'Spanish', 'ar': 'Arabic', 
                        'ja': 'Japanese', 'ko': 'Korean', 'th': 'Thai', 
                        'ro': 'Romanian', 'tr': 'Turkish', 'zh': 'Simplified Chinese',
                        'ur': 'Urdu'}

   
    src_language = longer_languages[in_lang]
    tar_language = longer_languages[out_lang]
            
    translated_text_labse =  (translate_cachable(in_text, PROMPT, src_language, tar_language, sys_name))

    return translated_text_labse  