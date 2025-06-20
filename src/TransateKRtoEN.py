# -*- coding: utf-8 -*-
import json
import logging
import shutil
import threading
import queue
import os, sys, io, zipfile, time, re, mimetypes, subprocess, tiktoken
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from collections import Counter
from unified_api_client import UnifiedClient, UnifiedClientError
import hashlib
import unicodedata
from difflib import SequenceMatcher
import unicodedata
import re
import time
from history_manager import HistoryManager
from chapter_splitter import ChapterSplitter
from image_translator import ImageTranslator
from typing import Dict, List, Tuple 
from txt_processor import TextFileProcessor

# =====================================================
# CONFIGURATION AND ENVIRONMENT MANAGEMENT
# =====================================================
class TranslationConfig:
    """Centralized configuration management"""
    def __init__(self):
        self.MODEL = os.getenv("MODEL", "gemini-1.5-flash")
        self.input_path = os.getenv("input_path", "default.epub")
        self.PROFILE_NAME = os.getenv("PROFILE_NAME", "korean").lower()
        self.CONTEXTUAL = os.getenv("CONTEXTUAL", "1") == "1"
        self.DELAY = int(os.getenv("SEND_INTERVAL_SECONDS", "2"))
        self.SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "").strip()
        self.REMOVE_AI_ARTIFACTS = os.getenv("REMOVE_AI_ARTIFACTS", "0") == "1"
        self.TEMP = float(os.getenv("TRANSLATION_TEMPERATURE", "0.3"))
        self.HIST_LIMIT = int(os.getenv("TRANSLATION_HISTORY_LIMIT", "20"))
        self.MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "8192"))
        self.EMERGENCY_RESTORE = os.getenv("EMERGENCY_PARAGRAPH_RESTORE", "1") == "1"
        self.BATCH_TRANSLATION = os.getenv("BATCH_TRANSLATION", "0") == "1"  
        self.BATCH_SIZE = int(os.getenv("BATCH_SIZE", "10"))
        self.ENABLE_IMAGE_TRANSLATION = os.getenv("ENABLE_IMAGE_TRANSLATION", "1") == "1"
        self.TRANSLATE_BOOK_TITLE = os.getenv("TRANSLATE_BOOK_TITLE", "1") == "1"
        self.DISABLE_ZERO_DETECTION = os.getenv("DISABLE_ZERO_DETECTION", "0") == "1"
        self.COMPREHENSIVE_EXTRACTION = os.getenv("COMPREHENSIVE_EXTRACTION", "0") == "1"
        self.DISABLE_AUTO_GLOSSARY = os.getenv("DISABLE_AUTO_GLOSSARY", "0") == "1"
        self.MANUAL_GLOSSARY = os.getenv("MANUAL_GLOSSARY")
        self.RETRY_TRUNCATED = os.getenv("RETRY_TRUNCATED", "0") == "1"
        self.RETRY_DUPLICATE_BODIES = os.getenv("RETRY_DUPLICATE_BODIES", "1") == "1"
        self.RETRY_TIMEOUT = os.getenv("RETRY_TIMEOUT", "0") == "1"
        self.CHUNK_TIMEOUT = int(os.getenv("CHUNK_TIMEOUT", "900"))
        self.MAX_RETRY_TOKENS = int(os.getenv("MAX_RETRY_TOKENS", "16384"))
        self.DUPLICATE_LOOKBACK_CHAPTERS = int(os.getenv("DUPLICATE_LOOKBACK_CHAPTERS", "3"))
        self.USE_ROLLING_SUMMARY = os.getenv("USE_ROLLING_SUMMARY", "0") == "1"
        self.ROLLING_SUMMARY_EXCHANGES = int(os.getenv("ROLLING_SUMMARY_EXCHANGES", "5"))
        self.ROLLING_SUMMARY_MODE = os.getenv("ROLLING_SUMMARY_MODE", "append")
        self.DUPLICATE_DETECTION_MODE = os.getenv("DUPLICATE_DETECTION_MODE", "basic")
        self.AI_HUNTER_THRESHOLD = int(os.getenv("AI_HUNTER_THRESHOLD", "75"))
        self.TRANSLATION_HISTORY_ROLLING = os.getenv("TRANSLATION_HISTORY_ROLLING", "0") == "1"
        self.API_KEY = (os.getenv("API_KEY") or 
                       os.getenv("OPENAI_API_KEY") or 
                       os.getenv("OPENAI_OR_Gemini_API_KEY") or
                       os.getenv("GEMINI_API_KEY"))

# =====================================================
# UNIFIED PATTERNS AND CONSTANTS
# =====================================================
class PatternManager:
    """Centralized pattern management"""
    
    CHAPTER_PATTERNS = [
        # English patterns
        (r'chapter[\s_-]*(\d+)', re.IGNORECASE, 'english_chapter'),
        (r'\bch\.?\s*(\d+)\b', re.IGNORECASE, 'english_ch'),
        (r'part[\s_-]*(\d+)', re.IGNORECASE, 'english_part'),
        (r'episode[\s_-]*(\d+)', re.IGNORECASE, 'english_episode'),
        # Chinese patterns
        (r'第\s*(\d+)\s*[章节話话回]', 0, 'chinese_chapter'),
        (r'第\s*([一二三四五六七八九十百千万]+)\s*[章节話话回]', 0, 'chinese_chapter_cn'),
        (r'(\d+)[章节話话回]', 0, 'chinese_short'),
        # Japanese patterns
        (r'第\s*(\d+)\s*話', 0, 'japanese_wa'),
        (r'第\s*(\d+)\s*章', 0, 'japanese_chapter'),
        (r'その\s*(\d+)', 0, 'japanese_sono'),
        (r'(\d+)話目', 0, 'japanese_wame'),
        # Korean patterns
        (r'제\s*(\d+)\s*[장화권부편]', 0, 'korean_chapter'),
        (r'(\d+)\s*[장화권부편]', 0, 'korean_short'),
        (r'에피소드\s*(\d+)', 0, 'korean_episode'),
        # Generic numeric patterns
        (r'^\s*(\d+)\s*[-–—.\:]', re.MULTILINE, 'generic_numbered'),
        (r'_(\d+)\.x?html?$', re.IGNORECASE, 'filename_number'),
        (r'/(\d+)\.x?html?$', re.IGNORECASE, 'path_number'),
        (r'(\d+)', 0, 'any_number'),
    ]
    
    FILENAME_EXTRACT_PATTERNS = [
        r'^(\d+)[_\.]',          # "0249_" or "0249."
        r'^(\d{3,4})$',          # "0249" (just numbers)
        r'No(\d+)Chapter',       # "No249Chapter"
        r'[Cc]hapter[_\s]*(\d+)', # "Chapter249" or "chapter_249"
        r'[Cc]h[_\s]*(\d+)',     # "ch249" or "CH 249"
        r'_(\d+)$',              # "text_249"
        r'-(\d+)$',              # "text-249"
        r'(\d+)$'                # "text249" (fallback: any number)
    ]
    
    CJK_HONORIFICS = {
        'korean': ['님', '씨', '선배', '형', '누나', '언니', '오빠', '선생님', '교수님', '사장님', '회장님'],
        'japanese': ['さん', 'ちゃん', '君', 'くん', '様', 'さま', '先生', 'せんせい', '殿', 'どの', '先輩', 'せんぱい'],
        'chinese': ['先生', '小姐', '夫人', '公子', '大人', '老师', '师父', '师傅', '同志', '同学'],
        'english': [
            ' san', ' chan', ' kun', ' sama', ' sensei', ' senpai', ' dono', ' shi', ' tan', ' chin',
            '-ssi', '-nim', '-ah', '-ya', '-hyung', '-hyungnim', '-oppa', '-unnie', '-noona', 
            '-sunbae', '-sunbaenim', '-hubae', '-seonsaeng', '-seonsaengnim', '-gun', '-yang',
            '-xiong', '-di', '-ge', '-gege', '-didi', '-jie', '-jiejie', '-meimei', '-shixiong',
            '-shidi', '-shijie', '-shimei', '-gongzi', '-guniang', '-xiaojie', '-daren', '-qianbei',
            '-daoyou', '-zhanglao', '-shibo', '-shishu', '-shifu', '-laoshi', '-xiansheng'
        ]
    }
    
    TITLE_PATTERNS = {
        'korean': [
            r'\b(왕|여왕|왕자|공주|황제|황후|대왕|대공|공작|백작|자작|남작|기사|장군|대장|원수|제독|함장|대신|재상|총리|대통령|시장|지사|검사|판사|변호사|의사|박사|교수|신부|목사|스님|도사)\b',
            r'\b(폐하|전하|각하|예하|님|대감|영감|나리|도련님|아가씨|부인|선생)\b'
        ],
        'japanese': [
            r'\b(王|女王|王子|姫|皇帝|皇后|天皇|皇太子|大王|大公|公爵|伯爵|子爵|男爵|騎士|将軍|大将|元帥|提督|艦長|大臣|宰相|総理|大統領|市長|知事|検事|裁判官|弁護士|医者|博士|教授|神父|牧師|僧侶|道士)\b',
            r'\b(陛下|殿下|閣下|猊下|様|大人|殿|卿|君|氏)\b'
        ],
        'chinese': [
            r'\b(王|女王|王子|公主|皇帝|皇后|大王|大公|公爵|伯爵|子爵|男爵|骑士|将军|大将|元帅|提督|舰长|大臣|宰相|总理|大总统|市长|知事|检察官|法官|律师|医生|博士|教授|神父|牧师|和尚|道士)\b',
            r'\b(陛下|殿下|阁下|大人|老爷|夫人|小姐|公子|少爷|姑娘|先生)\b'
        ],
        'english': [
            r'\b(King|Queen|Prince|Princess|Emperor|Empress|Duke|Duchess|Marquis|Marquess|Earl|Count|Countess|Viscount|Viscountess|Baron|Baroness|Knight|Lord|Lady|Sir|Dame|General|Admiral|Captain|Major|Colonel|Commander|Lieutenant|Sergeant|Minister|Chancellor|President|Mayor|Governor|Judge|Doctor|Professor|Father|Reverend|Master|Mistress)\b',
            r'\b(His|Her|Your|Their)\s+(Majesty|Highness|Grace|Excellency|Honor|Worship|Lordship|Ladyship)\b'
        ]
    }
    
    CHINESE_NUMS = {
        '一': 1, '二': 2, '三': 3, '四': 4, '五': 5,
        '六': 6, '七': 7, '八': 8, '九': 9, '十': 10,
        '十一': 11, '十二': 12, '十三': 13, '十四': 14, '十五': 15,
        '十六': 16, '十七': 17, '十八': 18, '十九': 19, '二十': 20,
        '二十一': 21, '二十二': 22, '二十三': 23, '二十四': 24, '二十五': 25,
        '三十': 30, '四十': 40, '五十': 50, '六十': 60,
        '七十': 70, '八十': 80, '九十': 90, '百': 100,
    }
    
    COMMON_WORDS = {
        '이', '그', '저', '우리', '너희', '자기', '당신', '여기', '거기', '저기',
        '오늘', '내일', '어제', '지금', '아까', '나중', '먼저', '다음', '마지막',
        '모든', '어떤', '무슨', '이런', '그런', '저런', '같은', '다른', '새로운',
        '하다', '있다', '없다', '되다', '하는', '있는', '없는', '되는',
        '것', '수', '때', '년', '월', '일', '시', '분', '초',
        '은', '는', '이', '가', '을', '를', '에', '의', '와', '과', '도', '만',
        '에서', '으로', '로', '까지', '부터', '에게', '한테', '께', '께서',
        'この', 'その', 'あの', 'どの', 'これ', 'それ', 'あれ', 'どれ',
        'わたし', 'あなた', 'かれ', 'かのじょ', 'わたしたち', 'あなたたち',
        'きょう', 'あした', 'きのう', 'いま', 'あとで', 'まえ', 'つぎ',
        'の', 'は', 'が', 'を', 'に', 'で', 'と', 'も', 'や', 'から', 'まで',
        '这', '那', '哪', '这个', '那个', '哪个', '这里', '那里', '哪里',
        '我', '你', '他', '她', '它', '我们', '你们', '他们', '她们',
        '今天', '明天', '昨天', '现在', '刚才', '以后', '以前', '后来',
        '的', '了', '在', '是', '有', '和', '与', '或', '但', '因为', '所以',
        '一', '二', '三', '四', '五', '六', '七', '八', '九', '十',
        '1', '2', '3', '4', '5', '6', '7', '8', '9', '0',
    }

# =====================================================
# CHUNK CONTEXT MANAGER (unchanged - already optimal)
# =====================================================
class ChunkContextManager:
    """Manage context within a chapter separate from history"""
    def __init__(self):
        self.current_chunks = []
        self.chapter_num = None
        self.chapter_title = None
        
    def start_chapter(self, chapter_num, chapter_title):
        """Start a new chapter context"""
        self.current_chunks = []
        self.chapter_num = chapter_num
        self.chapter_title = chapter_title
        
    def add_chunk(self, user_content, assistant_content, chunk_idx, total_chunks):
        """Add a chunk to the current chapter context"""
        self.current_chunks.append({
            "user": user_content,
            "assistant": assistant_content,
            "chunk_idx": chunk_idx,
            "total_chunks": total_chunks
        })
    
    def get_context_messages(self, limit=3):
        """Get last N chunks as messages for API context"""
        context = []
        for chunk in self.current_chunks[-limit:]:
            context.extend([
                {"role": "user", "content": chunk["user"]},
                {"role": "assistant", "content": chunk["assistant"]}
            ])
        return context
    
    def get_summary_for_history(self):
        """Create a summary representation for the history"""
        if not self.current_chunks:
            return None, None
            
        total_chunks = len(self.current_chunks)
        
        user_summary = f"[Chapter {self.chapter_num}: {self.chapter_title}]\n"
        user_summary += f"[{total_chunks} chunks processed]\n"
        if self.current_chunks:
            first_chunk = self.current_chunks[0]['user']
            if len(first_chunk) > 500:
                user_summary += first_chunk[:500] + "..."
            else:
                user_summary += first_chunk
        
        assistant_summary = f"[Chapter {self.chapter_num} Translation Complete]\n"
        assistant_summary += f"[Translated in {total_chunks} chunks]\n"
        if self.current_chunks:
            samples = []
            first_trans = self.current_chunks[0]['assistant']
            samples.append(f"Beginning: {first_trans[:200]}..." if len(first_trans) > 200 else f"Beginning: {first_trans}")
            
            if total_chunks > 2:
                mid_idx = total_chunks // 2
                mid_trans = self.current_chunks[mid_idx]['assistant']
                samples.append(f"Middle: {mid_trans[:200]}..." if len(mid_trans) > 200 else f"Middle: {mid_trans}")
            
            if total_chunks > 1:
                last_trans = self.current_chunks[-1]['assistant']
                samples.append(f"End: {last_trans[:200]}..." if len(last_trans) > 200 else f"End: {last_trans}")
            
            assistant_summary += "\n".join(samples)
        
        return user_summary, assistant_summary
    
    def clear(self):
        """Clear the current chapter context"""
        self.current_chunks = []
        self.chapter_num = None 
        self.chapter_title = None

# =====================================================
# UNIFIED UTILITIES
# =====================================================
class FileUtilities:
    """Utilities for file and path operations"""
    
    @staticmethod
    def extract_actual_chapter_number(chapter, patterns=None):
        """Extract actual chapter number from filename"""
        if patterns is None:
            patterns = PatternManager.FILENAME_EXTRACT_PATTERNS
        
        actual_num = None
        if chapter.get('original_basename'):
            basename = chapter['original_basename']
            for pattern in patterns:
                match = re.search(pattern, basename, re.IGNORECASE)
                if match:
                    actual_num = int(match.group(1))
                    break
        
        if actual_num is None:
            actual_num = chapter.get("num", 0)
        
        return actual_num
    
    @staticmethod
    def create_chapter_filename(chapter, actual_num=None):
        """Create consistent chapter filename"""
        if actual_num is None:
            actual_num = FileUtilities.extract_actual_chapter_number(chapter)
        
        if 'original_basename' in chapter and chapter['original_basename']:
            return f"response_{chapter['original_basename']}.html"
        else:
            safe_title = make_safe_filename(chapter['title'], actual_num)   
            if isinstance(actual_num, float):
                return f"response_{actual_num:06.1f}_{safe_title}.html"
            else:
                return f"response_{actual_num:03d}_{safe_title}.html"

# =====================================================
# UNIFIED PROGRESS MANAGER
# =====================================================
class ProgressManager:
    """Unified progress management"""
    
    def __init__(self, payloads_dir):
        self.payloads_dir = payloads_dir
        self.PROGRESS_FILE = os.path.join(payloads_dir, "translation_progress.json")
        self.prog = self._init_or_load()
        
    def _init_or_load(self):
        """Initialize or load progress tracking with improved structure"""
        if os.path.exists(self.PROGRESS_FILE):
            try:
                with open(self.PROGRESS_FILE, "r", encoding="utf-8") as pf:
                    prog = json.load(pf)
            except json.JSONDecodeError as e:
                print(f"⚠️ Warning: Progress file is corrupted: {e}")
                print("🔧 Attempting to fix JSON syntax...")
                
                try:
                    with open(self.PROGRESS_FILE, "r", encoding="utf-8") as pf:
                        content = pf.read()
                    
                    content = re.sub(r',\s*\]', ']', content)
                    content = re.sub(r',\s*\}', '}', content)
                    
                    prog = json.loads(content)
                    
                    with open(self.PROGRESS_FILE, "w", encoding="utf-8") as pf:
                        json.dump(prog, pf, ensure_ascii=False, indent=2)
                    print("✅ Successfully fixed and saved progress file")
                    
                except Exception as fix_error:
                    print(f"❌ Could not fix progress file: {fix_error}")
                    print("🔄 Creating backup and starting fresh...")
                    
                    backup_name = f"translation_progress_backup_{int(time.time())}.json"
                    backup_path = os.path.join(self.payloads_dir, backup_name)
                    try:
                        shutil.copy(self.PROGRESS_FILE, backup_path)
                        print(f"📁 Backup saved to: {backup_name}")
                    except:
                        pass
                    
                    prog = {
                        "chapters": {},
                        "content_hashes": {},
                        "chapter_chunks": {},
                        "version": "2.0"
                    }
            
            if "chapters" not in prog:
                prog["chapters"] = {}
                
                for idx in prog.get("completed", []):
                    prog["chapters"][str(idx)] = {
                        "status": "completed",
                        "timestamp": None
                    }
            
            if "content_hashes" not in prog:
                prog["content_hashes"] = {}
            if "chapter_chunks" not in prog:
                prog["chapter_chunks"] = {}
                
        else:
            prog = {
                "chapters": {},
                "content_hashes": {},
                "chapter_chunks": {},
                "image_chunks": {},
                "version": "2.1"
            }
        
        return prog
    
    def save(self):
        """Save progress to file"""
        try:
            self.prog["completed_list"] = []
            for chapter_key, chapter_info in self.prog.get("chapters", {}).items():
                if chapter_info.get("status") == "completed" and chapter_info.get("output_file"):
                    self.prog["completed_list"].append({
                        "num": chapter_info.get("chapter_num", 0),
                        "idx": chapter_info.get("chapter_idx", 0),
                        "title": f"Chapter {chapter_info.get('chapter_num', 0)}",
                        "file": chapter_info.get("output_file", ""),
                        "key": chapter_key
                    })
            
            if self.prog.get("completed_list"):
                self.prog["completed_list"].sort(key=lambda x: x["num"])
            
            temp_file = self.PROGRESS_FILE + '.tmp'
            with open(temp_file, "w", encoding="utf-8") as pf:
                json.dump(self.prog, pf, ensure_ascii=False, indent=2)
            
            if os.path.exists(self.PROGRESS_FILE):
                os.remove(self.PROGRESS_FILE)
            os.rename(temp_file, self.PROGRESS_FILE)
        except Exception as e:
            print(f"⚠️ Warning: Failed to save progress: {e}")
            temp_file = self.PROGRESS_FILE + '.tmp'
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass
    
    def update(self, chapter_idx, actual_num, content_hash, output_filename=None, status="completed"):
        """Update progress tracking after successful translation"""
        chapter_key = content_hash
        
        self.prog["chapters"][chapter_key] = {
            "chapter_idx": chapter_idx,
            "actual_num": actual_num,
            "content_hash": content_hash,
            "output_file": output_filename,
            "status": status,
            "timestamp": time.time(),
            "display_name": f"Chapter {actual_num}",  
            "file_basename": output_filename.replace('response_', '').replace('.html', '') if output_filename else None
        }
        
        if output_filename and status == "completed":
            self.prog["content_hashes"][content_hash] = {
                "chapter_idx": chapter_idx,
                "actual_num": actual_num,
                "output_file": output_filename
            }
    
    def check_chapter_status(self, chapter_idx, actual_num, content_hash, output_dir):
        """Check if a chapter needs translation with fallback"""
        chapter_key = content_hash
        
        if chapter_key in self.prog["chapters"]:
            chapter_info = self.prog["chapters"][chapter_key]
        else:
            old_key = str(chapter_idx)
            if old_key in self.prog["chapters"]:
                print(f"[PROGRESS] Using legacy index-based tracking for chapter {actual_num}")
                chapter_info = self.prog["chapters"][old_key]
                self.prog["chapters"][chapter_key] = chapter_info
                chapter_info["content_hash"] = content_hash
                del self.prog["chapters"][old_key]
            else:
                return True, None, None
        
        output_file = chapter_info.get("output_file")
        
        if output_file:
            output_path = os.path.join(output_dir, output_file)
            if not os.path.exists(output_path):
                print(f"⚠️ Output file missing for chapter {actual_num}: {output_file}")
                print(f"🔄 Chapter {actual_num} will be retranslated")
                return True, None, None
            else:
                try:
                    with open(output_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if len(content.strip()) < 100:
                            print(f"⚠️ Output file for chapter {actual_num} seems corrupted/empty")
                            return True, None, None
                except Exception as e:
                    print(f"⚠️ Error reading output file for chapter {actual_num}: {e}")
                    return True, None, None
                
                return False, f"Chapter {actual_num} already translated (file exists: {output_file})", output_file
        
        elif chapter_info.get("status") in ["in_progress", "file_deleted"]:
            return True, None, None
        
        if content_hash in self.prog["content_hashes"]:
            duplicate_info = self.prog["content_hashes"][content_hash]
            duplicate_idx = duplicate_info.get("chapter_idx")
            
            duplicate_key = None
            for key, info in self.prog["chapters"].items():
                if info.get("chapter_idx") == duplicate_idx:
                    duplicate_key = key
                    break
            if not duplicate_key:
                duplicate_key = str(duplicate_idx)
                
            if duplicate_key in self.prog["chapters"]:
                dup_chapter = self.prog["chapters"][duplicate_key]
                dup_output_file = dup_chapter.get("output_file")
                if dup_output_file:
                    dup_path = os.path.join(output_dir, dup_output_file)
                    if os.path.exists(dup_path):
                        try:
                            with open(dup_path, 'r', encoding='utf-8') as f:
                                dup_content = f.read()
                                if len(dup_content.strip()) < 100:
                                    return True, None, None
                        except:
                            return True, None, None
                        
                        return False, f"Chapter {actual_num} has same content as chapter {duplicate_info.get('actual_num')} (already translated)", None
                    else:
                        return True, None, None
        
        return True, None, None
    
    def cleanup_missing_files(self, output_dir):
        """Scan progress tracking and clean up any references to missing files"""
        cleaned_count = 0
        
        for chapter_key, chapter_info in list(self.prog["chapters"].items()):
            output_file = chapter_info.get("output_file")
            
            if output_file:
                output_path = os.path.join(output_dir, output_file)
                if not os.path.exists(output_path):
                    print(f"🧹 Found missing file for chapter {chapter_info.get('actual_num', chapter_key)}: {output_file}")
                    
                    chapter_info["status"] = "file_deleted"
                    chapter_info["output_file"] = None
                    
                    content_hash = chapter_info.get("content_hash")
                    if content_hash and content_hash in self.prog["content_hashes"]:
                        stored_info = self.prog["content_hashes"][content_hash]
                        if content_hash == chapter_key:
                            del self.prog["content_hashes"][content_hash]
                    
                    if chapter_key in self.prog.get("chapter_chunks", {}):
                        del self.prog["chapter_chunks"][chapter_key]
                    
                    cleaned_count += 1
        
        if cleaned_count > 0:
            print(f"🔄 Marked {cleaned_count} chapters for retranslation due to missing files")
    
    def migrate_to_content_hash(self, chapters):
        """Migrate old index-based progress to content-hash-based"""
        if not self.prog.get("version") or self.prog["version"] < "3.0":
            print("🔄 Migrating progress tracking to content-hash-based system...")
            
            old_chapters = self.prog.get("chapters", {})
            new_chapters = {}
            old_chunks = self.prog.get("chapter_chunks", {})
            new_chunks = {}
            
            idx_to_hash = {}
            for idx, chapter in enumerate(chapters):
                content_hash = chapter.get("content_hash") or get_content_hash(chapter["body"])
                idx_to_hash[str(idx)] = content_hash
            
            for old_key, chapter_info in old_chapters.items():
                if old_key in idx_to_hash:
                    new_key = idx_to_hash[old_key]
                    new_chapters[new_key] = chapter_info
                    chapter_info["chapter_idx"] = int(old_key)
            
            for old_key, chunk_info in old_chunks.items():
                if old_key in idx_to_hash:
                    new_key = idx_to_hash[old_key]
                    new_chunks[new_key] = chunk_info
            
            self.prog["chapters"] = new_chapters
            self.prog["chapter_chunks"] = new_chunks
            self.prog["version"] = "3.0"
            
            print(f"✅ Migrated {len(new_chapters)} chapters to new system")
    
    def get_stats(self, output_dir):
        """Get statistics about translation progress"""
        stats = {
            "total_tracked": len(self.prog["chapters"]),
            "completed": 0,
            "missing_files": 0,
            "in_progress": 0
        }
        
        for chapter_info in self.prog["chapters"].values():
            status = chapter_info.get("status")
            output_file = chapter_info.get("output_file")
            
            if status == "completed" and output_file:
                output_path = os.path.join(output_dir, output_file)
                if os.path.exists(output_path):
                    stats["completed"] += 1
                else:
                    stats["missing_files"] += 1
            elif status == "in_progress":
                stats["in_progress"] += 1
            elif status == "file_missing":
                stats["missing_files"] += 1
        
        return stats

# =====================================================
# UNIFIED CONTENT PROCESSOR
# =====================================================
class ContentProcessor:
    """Unified content processing"""
    
    @staticmethod
    def clean_ai_artifacts(text, remove_artifacts=True):
        """Remove AI response artifacts from text - but ONLY when enabled"""
        if not remove_artifacts:
            return text
        
        lines = text.split('\n', 2)
        
        if len(lines) < 2:
            return text
        
        first_line = lines[0].strip()
        
        if not first_line:
            if len(lines) > 1:
                first_line = lines[1].strip()
                if not first_line:
                    return text
                lines = lines[1:]
            else:
                return text
        
        ai_patterns = [
            r'^(?:Sure|Okay|Understood|Of course|Got it|Alright|Certainly|Here\'s|Here is)',
            r'^(?:I\'ll|I will|Let me) (?:translate|help|assist)',
            r'^(?:System|Assistant|AI|User|Human|Model)\s*:',
            r'^\[PART\s+\d+/\d+\]',
            r'^(?:Translation note|Note|Here\'s the translation|I\'ve translated)',
            r'^```(?:html)?',
            r'^<!DOCTYPE',
        ]
        
        for pattern in ai_patterns:
            if re.search(pattern, first_line, re.IGNORECASE):
                remaining_text = '\n'.join(lines[1:]) if len(lines) > 1 else ''
                
                if remaining_text.strip():
                    if (re.search(r'<h[1-6]', remaining_text, re.IGNORECASE) or 
                        re.search(r'Chapter\s+\d+', remaining_text, re.IGNORECASE) or
                        re.search(r'第\s*\d+\s*[章節話话回]', remaining_text) or
                        re.search(r'제\s*\d+\s*[장화]', remaining_text) or
                        len(remaining_text.strip()) > 100):
                        
                        print(f"✂️ Removed AI artifact: {first_line[:50]}...")
                        return remaining_text.lstrip()
        
        if first_line.lower() in ['html', 'text', 'content', 'translation', 'output']:
            remaining_text = '\n'.join(lines[1:]) if len(lines) > 1 else ''
            if remaining_text.strip():
                print(f"✂️ Removed single word artifact: {first_line}")
                return remaining_text.lstrip()
        
        return text
    
    @staticmethod
    def clean_memory_artifacts(text):
        """Remove any memory/summary artifacts that leaked into the translation"""
        text = re.sub(r'\[MEMORY\].*?\[END MEMORY\]', '', text, flags=re.DOTALL)
        
        lines = text.split('\n')
        cleaned_lines = []
        skip_next = False
        
        for line in lines:
            if any(marker in line for marker in ['[MEMORY]', '[END MEMORY]', 'Previous context summary:', 
                                                  'memory summary', 'context summary', '[Context]']):
                skip_next = True
                continue
            
            if skip_next and line.strip() == '':
                skip_next = False
                continue
                
            skip_next = False
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    @staticmethod
    def emergency_restore_paragraphs(text, original_html=None, verbose=True):
        """Emergency restoration when AI returns wall of text without proper paragraph tags"""
        def log(message):
            if verbose:
                print(message)
        
        if text.count('</p>') >= 3:
            return text
        
        if original_html:
            original_para_count = original_html.count('<p>')
            current_para_count = text.count('<p>')
            
            if current_para_count < original_para_count / 2:
                log(f"⚠️ Paragraph mismatch! Original: {original_para_count}, Current: {current_para_count}")
                log("🔧 Attempting emergency paragraph restoration...")
        
        if '</p>' not in text and len(text) > 300:
            log("❌ No paragraph tags found - applying emergency restoration")
            
            if '\n\n' in text:
                parts = text.split('\n\n')
                paragraphs = ['<p>' + part.strip() + '</p>' for part in parts if part.strip()]
                return '\n'.join(paragraphs)
            
            dialogue_pattern = r'(?<=[.!?])\s+(?=[""\u201c\u201d])'
            if re.search(dialogue_pattern, text):
                parts = re.split(dialogue_pattern, text)
                paragraphs = []
                for part in parts:
                    part = part.strip()
                    if part:
                        if not part.startswith('<p>'):
                            part = '<p>' + part
                        if not part.endswith('</p>'):
                            part = part + '</p>'
                        paragraphs.append(part)
                return '\n'.join(paragraphs)
            
            sentence_boundary = r'(?<=[.!?])\s+(?=[A-Z\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af])'
            sentences = re.split(sentence_boundary, text)
            
            if len(sentences) > 1:
                paragraphs = []
                current_para = []
                
                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                        
                    current_para.append(sentence)
                    
                    should_break = (
                        len(current_para) >= 3 or
                        sentence.rstrip().endswith(('"', '"', '"')) or
                        '* * *' in sentence or
                        '***' in sentence or
                        '---' in sentence
                    )
                    
                    if should_break:
                        para_text = ' '.join(current_para)
                        if not para_text.startswith('<p>'):
                            para_text = '<p>' + para_text
                        if not para_text.endswith('</p>'):
                            para_text = para_text + '</p>'
                        paragraphs.append(para_text)
                        current_para = []
                
                if current_para:
                    para_text = ' '.join(current_para)
                    if not para_text.startswith('<p>'):
                        para_text = '<p>' + para_text
                    if not para_text.endswith('</p>'):
                        para_text = para_text + '</p>'
                    paragraphs.append(para_text)
                
                result = '\n'.join(paragraphs)
                log(f"✅ Restored {len(paragraphs)} paragraphs from wall of text")
                return result
            
            words = text.split()
            if len(words) > 100:
                paragraphs = []
                words_per_para = max(100, len(words) // 10)
                
                for i in range(0, len(words), words_per_para):
                    chunk = ' '.join(words[i:i + words_per_para])
                    if chunk.strip():
                        paragraphs.append('<p>' + chunk.strip() + '</p>')
                
                return '\n'.join(paragraphs)
        
        elif '<p>' in text and text.count('<p>') < 3 and len(text) > 1000:
            log("⚠️ Very few paragraphs for long text - checking if more breaks needed")
            
            soup = BeautifulSoup(text, 'html.parser')
            existing_paras = soup.find_all('p')
            
            new_paragraphs = []
            for para in existing_paras:
                para_text = para.get_text()
                if len(para_text) > 500:
                    sentences = re.split(r'(?<=[.!?])\s+', para_text)
                    if len(sentences) > 5:
                        chunks = []
                        current = []
                        for sent in sentences:
                            current.append(sent)
                            if len(current) >= 3:
                                chunks.append('<p>' + ' '.join(current) + '</p>')
                                current = []
                        if current:
                            chunks.append('<p>' + ' '.join(current) + '</p>')
                        new_paragraphs.extend(chunks)
                    else:
                        new_paragraphs.append(str(para))
                else:
                    new_paragraphs.append(str(para))
            
            return '\n'.join(new_paragraphs)
        
        return text
    
    @staticmethod
    def get_content_hash(html_content):
        """Create a stable hash of content"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            for tag in soup(['script', 'style', 'meta', 'link']):
                tag.decompose()
            
            text_content = soup.get_text(separator=' ', strip=True)
            text_content = ' '.join(text_content.split())
            
            return hashlib.md5(text_content.encode('utf-8')).hexdigest()
            
        except Exception as e:
            print(f"[WARNING] Failed to create hash: {e}")
            return hashlib.md5(html_content.encode('utf-8')).hexdigest()
    
    @staticmethod
    def is_meaningful_text_content(html_content):
        """Check if chapter has meaningful text beyond just structure"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            soup_copy = BeautifulSoup(str(soup), 'html.parser')
            
            for img in soup_copy.find_all('img'):
                img.decompose()
            
            text_elements = soup_copy.find_all(['p', 'div', 'span'])
            text_content = ' '.join(elem.get_text(strip=True) for elem in text_elements)
            
            headers = soup_copy.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
            header_text = ' '.join(h.get_text(strip=True) for h in headers)
            
            if headers and len(text_content.strip()) > 1:
                return True
            
            if len(text_content.strip()) > 200:
                return True
            
            if len(header_text.strip()) > 100:
                return True
                
            return False
            
        except Exception as e:
            print(f"Warning: Error checking text content: {e}")
            return True

# =====================================================
# UNIFIED CHAPTER EXTRACTOR
# =====================================================
class ChapterExtractor:
    """Unified chapter extraction"""
    
    def __init__(self):
        self.pattern_manager = PatternManager()
        
    def extract_chapters(self, zf, output_dir):
        """Extract chapters and all resources from EPUB"""
        print("🚀 Starting comprehensive EPUB extraction...")
        print("✅ Using enhanced extraction with full resource handling")
        
        extracted_resources = self._extract_all_resources(zf, output_dir)
        
        metadata_path = os.path.join(output_dir, 'metadata.json')
        if os.path.exists(metadata_path):
            print("📋 Loading existing metadata...")
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        else:
            print("📋 Extracting fresh metadata...")
            metadata = self._extract_epub_metadata(zf)
            print(f"📋 Extracted metadata: {list(metadata.keys())}")
        
        chapters, detected_language = self._extract_chapters_universal(zf)
        
        if not chapters:
            print("❌ No chapters could be extracted!")
            return []
        
        chapters_info_path = os.path.join(output_dir, 'chapters_info.json')
        chapters_info = []
        
        for c in chapters:
            info = {
                'num': c['num'],
                'title': c['title'],
                'original_filename': c.get('filename', ''),
                'has_images': c.get('has_images', False),
                'image_count': c.get('image_count', 0),
                'text_length': c.get('file_size', len(c.get('body', ''))),
                'detection_method': c.get('detection_method', 'unknown'),
                'content_hash': c.get('content_hash', '')
            }
            
            if c.get('has_images'):
                try:
                    soup = BeautifulSoup(c.get('body', ''), 'html.parser')
                    images = soup.find_all('img')
                    info['images'] = [img.get('src', '') for img in images]
                except:
                    info['images'] = []
            
            chapters_info.append(info)
        
        with open(chapters_info_path, 'w', encoding='utf-8') as f:
            json.dump(chapters_info, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Saved detailed chapter info to: chapters_info.json")
        
        metadata.update({
            'chapter_count': len(chapters),
            'detected_language': detected_language,
            'extracted_resources': extracted_resources,
            'extraction_summary': {
                'total_chapters': len(chapters),
                'chapter_range': f"{chapters[0]['num']}-{chapters[-1]['num']}",
                'resources_extracted': sum(len(files) for files in extracted_resources.values())
            }
        })
        
        metadata['chapter_titles'] = {
            str(c['num']): c['title'] for c in chapters
        }
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Saved comprehensive metadata to: {metadata_path}")
        
        self._create_extraction_report(output_dir, metadata, chapters, extracted_resources)
        self._log_extraction_summary(chapters, extracted_resources, detected_language)
        
        print("🔍 VERIFICATION: Comprehensive chapter extraction completed successfully")
        
        return chapters
    
    def _extract_all_resources(self, zf, output_dir):
        """Extract all resources with duplicate prevention"""
        extracted_resources = {
            'css': [],
            'fonts': [],
            'images': [],
            'epub_structure': [],
            'other': []
        }
        
        extraction_marker = os.path.join(output_dir, '.resources_extracted')
        if os.path.exists(extraction_marker):
            print("📦 Resources already extracted, skipping resource extraction...")
            return self._count_existing_resources(output_dir, extracted_resources)
        
        self._cleanup_old_resources(output_dir)
        
        for resource_type in ['css', 'fonts', 'images']:
            os.makedirs(os.path.join(output_dir, resource_type), exist_ok=True)
        
        print(f"📦 Extracting all resources from EPUB...")
        
        for file_path in zf.namelist():
            if file_path.endswith('/') or not os.path.basename(file_path):
                continue
                
            try:
                file_data = zf.read(file_path)
                resource_info = self._categorize_resource(file_path, os.path.basename(file_path))
                
                if resource_info:
                    resource_type, target_dir, safe_filename = resource_info
                    target_path = os.path.join(output_dir, target_dir, safe_filename) if target_dir else os.path.join(output_dir, safe_filename)
                    
                    with open(target_path, 'wb') as f:
                        f.write(file_data)
                    
                    extracted_resources[resource_type].append(safe_filename)
                    
                    if resource_type == 'epub_structure':
                        print(f"   📋 Extracted EPUB structure: {safe_filename}")
                    else:
                        print(f"   📄 Extracted {resource_type}: {safe_filename}")
                    
            except Exception as e:
                print(f"[WARNING] Failed to extract {file_path}: {e}")
        
        with open(extraction_marker, 'w') as f:
            f.write(f"Resources extracted at {time.time()}")
        
        self._validate_critical_files(output_dir, extracted_resources)
        
        return extracted_resources
    
    def _extract_chapters_universal(self, zf, comprehensive_mode=None):
        """Universal chapter extraction with mode toggle"""
        if comprehensive_mode is None:
            comprehensive_mode = os.getenv("COMPREHENSIVE_EXTRACTION", "0") == "1"
        
        chapters = []
        sample_texts = []
        
        html_files = []
        for name in zf.namelist():
            if name.lower().endswith(('.xhtml', '.html', '.htm')):
                if not comprehensive_mode:
                    lower_name = name.lower()
                    if any(skip in lower_name for skip in [
                        'nav', 'toc', 'contents', 'cover', 'title', 'index',
                        'copyright', 'acknowledgment', 'dedication'
                    ]):
                        continue
                else:
                    skip_keywords = ['nav.', 'toc.', 'contents.', 'copyright.', 'cover.']
                    basename = os.path.basename(name.lower())
                    should_skip = False
                    for skip in skip_keywords:
                        if basename == skip + 'xhtml' or basename == skip + 'html' or basename == skip + 'htm':
                            should_skip = True
                            break
                    if should_skip:
                        print(f"[SKIP] Navigation/TOC file: {name}")
                        continue
                
                html_files.append(name)
        
        print(f"📚 Found {len(html_files)} {'HTML files' if comprehensive_mode else 'potential content files'} in EPUB")
        
        content_hashes = {}
        seen_chapters = {}
        file_size_groups = {}
        h1_count = 0
        h2_count = 0
        
        chapter_num = 0
        actual_num = 0
        
        for idx, file_path in enumerate(sorted(html_files)):
            try:
                file_data = zf.read(file_path)
                
                html_content = None
                for encoding in ['utf-8', 'utf-16', 'gb18030', 'shift_jis', 'euc-kr', 'gbk', 'big5']:
                    try:
                        html_content = file_data.decode(encoding)
                        break
                    except UnicodeDecodeError:
                        continue
                
                if not html_content:
                    print(f"[WARNING] Could not decode {file_path}")
                    continue
                
                soup = BeautifulSoup(html_content, 'html.parser')
                
                if soup.body:
                    content_html = str(soup.body)
                    content_text = soup.body.get_text(strip=True)
                else:
                    content_html = html_content
                    content_text = soup.get_text(strip=True)
                
                if comprehensive_mode:
                    actual_chapter_num = None
                    filename_base = os.path.basename(file_path)
                    
                    print(f"[EXTRACT DEBUG] Trying to extract number from: {filename_base}")

                    for pattern in self.pattern_manager.FILENAME_EXTRACT_PATTERNS:
                        match = re.search(pattern, filename_base, re.IGNORECASE)
                        if match:
                            actual_chapter_num = int(match.group(1))
                            break

                    if actual_chapter_num is not None:
                        chapter_num = actual_chapter_num
                    else:
                        chapter_num += 1
                else:
                    h1_tags = soup.find_all('h1')
                    h2_tags = soup.find_all('h2')
                    if h1_tags:
                        h1_count += 1
                    if h2_tags:
                        h2_count += 1
                
                images = soup.find_all('img')
                has_images = len(images) > 0
                
                is_image_only_chapter = has_images and len(content_text.strip()) < 500
                
                if not comprehensive_mode:
                    if len(content_text.strip()) < 10 and not has_images:
                        print(f"[DEBUG] Skipping empty file: {file_path} (no text, no images)")
                        continue
                    
                    if is_image_only_chapter:
                        print(f"[DEBUG] Image-only chapter detected: {file_path} ({len(images)} images, {len(content_text)} chars)")
                
                content_hash = ContentProcessor.get_content_hash(content_html)
                
                if not comprehensive_mode:
                    file_size = len(content_text)
                    if file_size not in file_size_groups:
                        file_size_groups[file_size] = []
                    file_size_groups[file_size].append(file_path)
                    
                    if content_hash in content_hashes:
                        duplicate_info = content_hashes[content_hash]
                        print(f"[DEBUG] Skipping duplicate content: {file_path} (matches {duplicate_info['filename']})")
                        continue
                    
                    if len(sample_texts) < 5:
                        sample_texts.append(content_text[:1000])
                
                if not comprehensive_mode:
                    chapter_num, chapter_title, detection_method = self._extract_chapter_info(
                        soup, file_path, content_text, html_content
                    )
                    
                    if chapter_num is None:
                        chapter_num = len(chapters) + 1
                        while chapter_num in seen_chapters:
                            chapter_num += 1
                        detection_method = "sequential_fallback"
                        print(f"[DEBUG] No chapter number found in {file_path}, assigning: {chapter_num}")
                    
                    if chapter_num in seen_chapters:
                        existing_info = seen_chapters[chapter_num]
                        existing_hash = existing_info['content_hash']
                        
                        if existing_hash != content_hash:
                            original_num = chapter_num
                            while chapter_num in seen_chapters:
                                chapter_num += 1
                            print(f"[WARNING] Chapter {original_num} already exists with different content")
                            print(f"[INFO] Reassigning {file_path} to chapter {chapter_num}")
                            detection_method += "_reassigned"
                        else:
                            print(f"[DEBUG] Skipping duplicate chapter {chapter_num}: {file_path}")
                            continue
                else:
                    chapter_title = None
                    if soup.title and soup.title.string:
                        chapter_title = soup.title.string.strip()
                    
                    if not chapter_title:
                        for header_tag in ['h1', 'h2', 'h3']:
                            header = soup.find(header_tag)
                            if header:
                                chapter_title = header.get_text(strip=True)
                                break
                    
                    if not chapter_title:
                        chapter_title = os.path.splitext(os.path.basename(file_path))[0]
                    
                    detection_method = "comprehensive_sequential"
                
                if actual_num is None:
                    actual_num = chapter_num
                
                chapter_info = {
                    "num": actual_num,
                    "title": chapter_title or f"Chapter {actual_num}",
                    "body": content_html,
                    "filename": file_path,
                    "original_basename": os.path.splitext(os.path.basename(file_path))[0],
                    "content_hash": content_hash,
                    "detection_method": detection_method,
                    "file_size": len(content_text),
                    "has_images": has_images,
                    "image_count": len(images),
                    "is_empty": len(content_text.strip()) == 0,
                    "is_image_only": is_image_only_chapter
                }
                
                if not comprehensive_mode:
                    chapter_info["language_sample"] = content_text[:500]
                
                chapters.append(chapter_info)
                
                if not comprehensive_mode:
                    content_hashes[content_hash] = {
                        'filename': file_path,
                        'chapter_num': chapter_num
                    }
                    seen_chapters[chapter_num] = {
                        'content_hash': content_hash,
                        'filename': file_path
                    }
                
                if comprehensive_mode:
                    if is_image_only_chapter:
                        print(f"[{actual_num:04d}] 📸 Image-only chapter: {chapter_title} ({len(images)} images)")
                    elif len(images) > 0 and len(content_text) >= 500:
                        print(f"[{actual_num:04d}] 📖📸 Mixed chapter: {chapter_title} ({len(content_text)} chars, {len(images)} images)")
                    elif len(content_text) < 50:
                        print(f"[{actual_num:04d}] 📄 Empty/placeholder: {chapter_title}")
                    else:
                        print(f"[{actual_num:04d}] 📖 Text chapter: {chapter_title} ({len(content_text)} chars)")
                else:
                    if is_image_only_chapter:
                        print(f"[DEBUG] ✅ Chapter {chapter_num}: {chapter_title[:50]}... (IMAGE-ONLY, {detection_method})")
                    else:
                        print(f"[DEBUG] ✅ Chapter {chapter_num}: {chapter_title[:50]}... ({detection_method})")
                    
            except Exception as e:
                print(f"[ERROR] Failed to process {file_path}: {e}")
                continue
        
        chapters.sort(key=lambda x: x["num"])
        
        if not comprehensive_mode and h2_count > h1_count:
            print(f"📊 Note: This EPUB uses primarily <h2> tags ({h2_count} files) vs <h1> tags ({h1_count} files)")
        
        combined_sample = ' '.join(sample_texts) if not comprehensive_mode else ''
        detected_language = self._detect_content_language(combined_sample) if combined_sample else 'unknown'
        
        if chapters:
            self._print_extraction_summary(chapters, detected_language, comprehensive_mode, 
                                         h1_count if not comprehensive_mode else 0, 
                                         h2_count if not comprehensive_mode else 0,
                                         file_size_groups if not comprehensive_mode else {})
        
        return chapters, detected_language
    
    def _extract_chapter_info(self, soup, file_path, content_text, html_content):
        """Extract chapter number and title from various sources"""
        chapter_num = None
        chapter_title = None
        detection_method = None
        
        # Try filename first
        for pattern, flags, method in self.pattern_manager.CHAPTER_PATTERNS:
            if method.endswith('_number'):
                match = re.search(pattern, file_path, flags)
                if match:
                    try:
                        num_str = match.group(1)
                        if num_str.isdigit():
                            chapter_num = int(num_str)
                            detection_method = f"filename_{method}"
                            break
                        elif method == 'chinese_chapter_cn':
                            converted = self._convert_chinese_number(num_str)
                            if converted:
                                chapter_num = converted
                                detection_method = f"filename_{method}"
                                break
                    except (ValueError, IndexError):
                        continue
        
        # Try content if not found in filename
        if not chapter_num:
            # Check title tag
            if soup.title and soup.title.string:
                title_text = soup.title.string.strip()
                chapter_num, detection_method = self._extract_from_text(title_text, "title")
                if chapter_num:
                    chapter_title = title_text
            
            # Check headers
            if not chapter_num:
                for header_tag in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                    headers = soup.find_all(header_tag)
                    for header in headers:
                        header_text = header.get_text(strip=True)
                        if header_text:
                            chapter_num, method = self._extract_from_text(header_text, f"header_{header_tag}")
                            if chapter_num:
                                chapter_title = header_text
                                detection_method = method
                                break
                    if chapter_num:
                        break
            
            # Check first paragraphs
            if not chapter_num:
                first_elements = soup.find_all(['p', 'div'])[:5]
                for elem in first_elements:
                    elem_text = elem.get_text(strip=True)
                    if elem_text:
                        chapter_num, detection_method = self._extract_from_text(elem_text, "content")
                        if chapter_num:
                            break
            
            # Final fallback to filename
            if not chapter_num:
                filename_base = os.path.basename(file_path)
                for pattern in self.pattern_manager.FILENAME_EXTRACT_PATTERNS:
                    match = re.search(pattern, filename_base, re.IGNORECASE)
                    if match:
                        chapter_num = int(match.group(1))
                        detection_method = "filename_number"
                        break
        
        # Extract title if not already found
        if not chapter_title:
            if soup.title and soup.title.string:
                chapter_title = soup.title.string.strip()
            else:
                for header_tag in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                    header = soup.find(header_tag)
                    if header:
                        chapter_title = header.get_text(strip=True)
                        break
            
            if not chapter_title:
                chapter_title = f"Chapter {chapter_num}" if chapter_num else None
        
        chapter_title = re.sub(r'\s+', ' ', chapter_title).strip() if chapter_title else None
        
        return chapter_num, chapter_title, detection_method
    
    def _extract_from_text(self, text, source_type):
        """Extract chapter number from text using patterns"""
        for pattern, flags, method in self.pattern_manager.CHAPTER_PATTERNS:
            if method.endswith('_number'):
                continue
            match = re.search(pattern, text, flags)
            if match:
                try:
                    num_str = match.group(1)
                    if num_str.isdigit():
                        return int(num_str), f"{source_type}_{method}"
                    elif method == 'chinese_chapter_cn':
                        converted = self._convert_chinese_number(num_str)
                        if converted:
                            return converted, f"{source_type}_{method}"
                except (ValueError, IndexError):
                    continue
        return None, None
    
    def _convert_chinese_number(self, cn_num):
        """Convert Chinese number to integer"""
        if cn_num in self.pattern_manager.CHINESE_NUMS:
            return self.pattern_manager.CHINESE_NUMS[cn_num]
        
        if '十' in cn_num:
            parts = cn_num.split('十')
            if len(parts) == 2:
                tens = self.pattern_manager.CHINESE_NUMS.get(parts[0], 1) if parts[0] else 1
                ones = self.pattern_manager.CHINESE_NUMS.get(parts[1], 0) if parts[1] else 0
                return tens * 10 + ones
        
        return None
    
    def _detect_content_language(self, text_sample):
        """Detect the primary language of content"""
        scripts = {
            'korean': 0,
            'japanese_hiragana': 0,
            'japanese_katakana': 0,
            'chinese': 0,
            'latin': 0
        }
        
        for char in text_sample:
            code = ord(char)
            if 0xAC00 <= code <= 0xD7AF:
                scripts['korean'] += 1
            elif 0x3040 <= code <= 0x309F:
                scripts['japanese_hiragana'] += 1
            elif 0x30A0 <= code <= 0x30FF:
                scripts['japanese_katakana'] += 1
            elif 0x4E00 <= code <= 0x9FFF:
                scripts['chinese'] += 1
            elif 0x0020 <= code <= 0x007F:
                scripts['latin'] += 1
        
        total_cjk = scripts['korean'] + scripts['japanese_hiragana'] + scripts['japanese_katakana'] + scripts['chinese']
        
        if scripts['korean'] > total_cjk * 0.3:
            return 'korean'
        elif scripts['japanese_hiragana'] + scripts['japanese_katakana'] > total_cjk * 0.2:
            return 'japanese'
        elif scripts['chinese'] > total_cjk * 0.3:
            return 'chinese'
        elif scripts['latin'] > len(text_sample) * 0.7:
            return 'english'
        else:
            return 'unknown'
    
    def _print_extraction_summary(self, chapters, detected_language, comprehensive_mode, h1_count, h2_count, file_size_groups):
        """Print extraction summary"""
        print(f"\n📊 Chapter Extraction Summary:")
        print(f"   • Total chapters extracted: {len(chapters)}")
        print(f"   • Chapter range: {chapters[0]['num']} to {chapters[-1]['num']}")
        print(f"   • Detected language: {detected_language}")
        
        if not comprehensive_mode:
            print(f"   • Primary header type: {'<h2>' if h2_count > h1_count else '<h1>'}")
        
        image_only_count = sum(1 for c in chapters if c.get('is_image_only', False))
        text_only_count = sum(1 for c in chapters if not c.get('has_images', False) and c.get('file_size', 0) >= 500)
        mixed_count = sum(1 for c in chapters if c.get('has_images', False) and c.get('file_size', 0) >= 500)
        
        print(f"   • Text-only chapters: {text_only_count}")
        print(f"   • Image-only chapters: {image_only_count}")
        print(f"   • Mixed content chapters: {mixed_count}")
        
        expected_chapters = set(range(chapters[0]['num'], chapters[-1]['num'] + 1))
        actual_chapters = set(c['num'] for c in chapters)
        missing = expected_chapters - actual_chapters
        if missing:
            print(f"   ⚠️ Missing chapter numbers: {sorted(missing)}")
        
        if not comprehensive_mode:
            method_stats = Counter(c['detection_method'] for c in chapters)
            print(f"   📈 Detection methods used:")
            for method, count in method_stats.most_common():
                print(f"      • {method}: {count} chapters")
            
            large_groups = [size for size, files in file_size_groups.items() if len(files) > 1]
            if large_groups:
                print(f"   ⚠️ Found {len(large_groups)} file size groups with potential duplicates")
        else:
            empty_chapters = sum(1 for c in chapters if c.get('is_empty'))
            print(f"   • Empty/placeholder: {empty_chapters}")
    
    def _extract_epub_metadata(self, zf):
        """Extract comprehensive metadata from EPUB file"""
        meta = {}
        try:
            for name in zf.namelist():
                if name.lower().endswith('.opf'):
                    opf_content = zf.read(name)
                    soup = BeautifulSoup(opf_content, 'xml')
                    
                    for tag in ['title', 'creator', 'language', 'publisher', 'date', 'subject']:
                        element = soup.find(tag)
                        if element:
                            meta[tag] = element.get_text(strip=True)
                    
                    description = soup.find('description')
                    if description:
                        meta['description'] = description.get_text(strip=True)
                    
                    meta_tags = soup.find_all('meta')
                    for meta_tag in meta_tags:
                        name = meta_tag.get('name', '').lower()
                        content = meta_tag.get('content', '')
                        if 'series' in name and content:
                            meta['series'] = content
                        elif 'calibre:series' in name and content:
                            meta['series'] = content
                    
                    break
        except Exception as e:
            print(f"[WARNING] Failed to extract metadata: {e}")
        
        return meta
    
    def _categorize_resource(self, file_path, file_name):
        """Categorize a file and return (resource_type, target_dir, safe_filename)"""
        file_path_lower = file_path.lower()
        file_name_lower = file_name.lower()
        
        if file_path_lower.endswith('.css'):
            return 'css', 'css', sanitize_resource_filename(file_name)
        elif file_path_lower.endswith(('.ttf', '.otf', '.woff', '.woff2', '.eot')):
            return 'fonts', 'fonts', sanitize_resource_filename(file_name)
        elif file_path_lower.endswith(('.jpg', '.jpeg', '.png', '.gif', '.svg', '.bmp', '.webp')):
            return 'images', 'images', sanitize_resource_filename(file_name)
        elif (file_path_lower.endswith(('.opf', '.ncx')) or 
              file_name_lower == 'container.xml' or
              'container.xml' in file_path_lower):
            if 'container.xml' in file_path_lower:
                safe_filename = 'container.xml'
            else:
                safe_filename = file_name
            return 'epub_structure', None, safe_filename
        elif file_path_lower.endswith(('.js', '.xml', '.txt')):
            return 'other', None, sanitize_resource_filename(file_name)
        
        return None
    
    def _cleanup_old_resources(self, output_dir):
        """Clean up old resource directories and EPUB structure files"""
        print("🧹 Cleaning up any existing resource directories...")
        
        cleanup_success = True
        
        for resource_type in ['css', 'fonts', 'images']:
            resource_dir = os.path.join(output_dir, resource_type)
            if os.path.exists(resource_dir):
                try:
                    shutil.rmtree(resource_dir)
                    print(f"   🗑️ Removed old {resource_type} directory")
                except PermissionError as e:
                    print(f"   ⚠️ Cannot remove {resource_type} directory (permission denied) - will merge with existing files")
                    cleanup_success = False
                except Exception as e:
                    print(f"   ⚠️ Error removing {resource_type} directory: {e} - will merge with existing files")
                    cleanup_success = False
        
        epub_structure_files = ['container.xml', 'content.opf', 'toc.ncx']
        for epub_file in epub_structure_files:
            input_path = os.path.join(output_dir, epub_file)
            if os.path.exists(input_path):
                try:
                    os.remove(input_path)
                    print(f"   🗑️ Removed old {epub_file}")
                except PermissionError:
                    print(f"   ⚠️ Cannot remove {epub_file} (permission denied) - will use existing file")
                except Exception as e:
                    print(f"   ⚠️ Error removing {epub_file}: {e}")
        
        try:
            for file in os.listdir(output_dir):
                if file.lower().endswith(('.opf', '.ncx')):
                    file_path = os.path.join(output_dir, file)
                    try:
                        os.remove(file_path)
                        print(f"   🗑️ Removed old EPUB file: {file}")
                    except PermissionError:
                        print(f"   ⚠️ Cannot remove {file} (permission denied)")
                    except Exception as e:
                        print(f"   ⚠️ Error removing {file}: {e}")
        except Exception as e:
            print(f"⚠️ Error scanning for EPUB files: {e}")
        
        if not cleanup_success:
            print("⚠️ Some cleanup operations failed due to file permissions")
            print("   The program will continue and merge with existing files")
        
        return cleanup_success
    
    def _count_existing_resources(self, output_dir, extracted_resources):
        """Count existing resources when skipping extraction"""
        for resource_type in ['css', 'fonts', 'images', 'epub_structure']:
            if resource_type == 'epub_structure':
                epub_files = []
                for file in ['container.xml', 'content.opf', 'toc.ncx']:
                    if os.path.exists(os.path.join(output_dir, file)):
                        epub_files.append(file)
                try:
                    for file in os.listdir(output_dir):
                        if file.lower().endswith(('.opf', '.ncx')) and file not in epub_files:
                            epub_files.append(file)
                except:
                    pass
                extracted_resources[resource_type] = epub_files
            else:
                resource_dir = os.path.join(output_dir, resource_type)
                if os.path.exists(resource_dir):
                    try:
                        files = [f for f in os.listdir(resource_dir) if os.path.isfile(os.path.join(resource_dir, f))]
                        extracted_resources[resource_type] = files
                    except:
                        extracted_resources[resource_type] = []
        
        total_existing = sum(len(files) for files in extracted_resources.values())
        print(f"✅ Found {total_existing} existing resource files")
        return extracted_resources
    
    def _validate_critical_files(self, output_dir, extracted_resources):
        """Validate that critical EPUB files were extracted"""
        total_extracted = sum(len(files) for files in extracted_resources.values())
        print(f"✅ Extracted {total_extracted} resource files:")
        
        for resource_type, files in extracted_resources.items():
            if files:
                if resource_type == 'epub_structure':
                    print(f"   • EPUB Structure: {len(files)} files")
                    for file in files:
                        print(f"     - {file}")
                else:
                    print(f"   • {resource_type.title()}: {len(files)} files")
        
        critical_files = ['container.xml']
        missing_critical = [f for f in critical_files if not os.path.exists(os.path.join(output_dir, f))]
        
        if missing_critical:
            print(f"⚠️ WARNING: Missing critical EPUB files: {missing_critical}")
            print("   This may prevent proper EPUB reconstruction!")
        else:
            print("✅ All critical EPUB structure files extracted successfully")
        
        opf_files = [f for f in extracted_resources['epub_structure'] if f.lower().endswith('.opf')]
        if not opf_files:
            print("⚠️ WARNING: No OPF file found! This will prevent EPUB reconstruction.")
        else:
            print(f"✅ Found OPF file(s): {opf_files}")
    
    def _create_extraction_report(self, output_dir, metadata, chapters, extracted_resources):
        """Create comprehensive extraction report with HTML file tracking"""
        report_path = os.path.join(output_dir, 'extraction_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("EPUB Extraction Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("METADATA:\n")
            for key, value in metadata.items():
                if key not in ['chapter_titles', 'extracted_resources']:
                    f.write(f"  {key}: {value}\n")
            
            f.write(f"\nCHAPTERS ({len(chapters)}):\n")
            
            text_chapters = []
            image_only_chapters = []
            mixed_chapters = []
            
            for chapter in chapters:
                if chapter.get('has_images') and chapter.get('file_size', 0) < 500:
                    image_only_chapters.append(chapter)
                elif chapter.get('has_images') and chapter.get('file_size', 0) >= 500:
                    mixed_chapters.append(chapter)
                else:
                    text_chapters.append(chapter)
            
            if text_chapters:
                f.write(f"\n  TEXT CHAPTERS ({len(text_chapters)}):\n")
                for c in text_chapters:
                    f.write(f"    {c['num']:3d}. {c['title']} ({c['detection_method']})\n")
                    if c.get('original_html_file'):
                        f.write(f"         → {c['original_html_file']}\n")
            
            if image_only_chapters:
                f.write(f"\n  IMAGE-ONLY CHAPTERS ({len(image_only_chapters)}):\n")
                for c in image_only_chapters:
                    f.write(f"    {c['num']:3d}. {c['title']} (images: {c.get('image_count', 0)})\n")
                    if c.get('original_html_file'):
                        f.write(f"         → {c['original_html_file']}\n")
                    if 'body' in c:
                        try:
                            soup = BeautifulSoup(c['body'], 'html.parser')
                            images = soup.find_all('img')
                            for img in images[:3]:
                                src = img.get('src', 'unknown')
                                f.write(f"         • Image: {src}\n")
                            if len(images) > 3:
                                f.write(f"         • ... and {len(images) - 3} more images\n")
                        except:
                            pass
            
            if mixed_chapters:
                f.write(f"\n  MIXED CONTENT CHAPTERS ({len(mixed_chapters)}):\n")
                for c in mixed_chapters:
                    f.write(f"    {c['num']:3d}. {c['title']} (text: {c.get('file_size', 0)} chars, images: {c.get('image_count', 0)})\n")
                    if c.get('original_html_file'):
                        f.write(f"         → {c['original_html_file']}\n")
            
            f.write(f"\nRESOURCES EXTRACTED:\n")
            for resource_type, files in extracted_resources.items():
                if files:
                    if resource_type == 'epub_structure':
                        f.write(f"  EPUB Structure: {len(files)} files\n")
                        for file in files:
                            f.write(f"    - {file}\n")
                    else:
                        f.write(f"  {resource_type.title()}: {len(files)} files\n")
                        for file in files[:5]:
                            f.write(f"    - {file}\n")
                        if len(files) > 5:
                            f.write(f"    ... and {len(files) - 5} more\n")
            
            f.write(f"\nHTML FILES WRITTEN:\n")
            html_files_written = metadata.get('html_files_written', 0)
            f.write(f"  Total: {html_files_written} files\n")
            f.write(f"  Location: Main directory and 'originals' subdirectory\n")
            
            f.write(f"\nPOTENTIAL ISSUES:\n")
            issues = []
            
            if image_only_chapters:
                issues.append(f"  • {len(image_only_chapters)} chapters contain only images (may need OCR)")
            
            missing_html = sum(1 for c in chapters if not c.get('original_html_file'))
            if missing_html > 0:
                issues.append(f"  • {missing_html} chapters failed to write HTML files")
            
            if not extracted_resources.get('epub_structure'):
                issues.append("  • No EPUB structure files found (may affect reconstruction)")
            
            if not issues:
                f.write("  None detected - extraction appears successful!\n")
            else:
                for issue in issues:
                    f.write(issue + "\n")
        
        print(f"📄 Saved extraction report to: {report_path}")
    
    def _log_extraction_summary(self, chapters, extracted_resources, detected_language, html_files_written=0):
        """Log final extraction summary with HTML file information"""
        print(f"\n✅ Comprehensive extraction complete!")
        print(f"   📚 Chapters: {len(chapters)}")
        print(f"   📄 HTML files written: {html_files_written}")
        print(f"   🎨 Resources: {sum(len(files) for files in extracted_resources.values())}")
        print(f"   🌍 Language: {detected_language}")
        
        image_only_count = sum(1 for c in chapters if c.get('has_images') and c.get('file_size', 0) < 500)
        if image_only_count > 0:
            print(f"   📸 Image-only chapters: {image_only_count}")
        
        epub_files = extracted_resources.get('epub_structure', [])
        if epub_files:
            print(f"   📋 EPUB Structure: {len(epub_files)} files ({', '.join(epub_files)})")
        else:
            print(f"   ⚠️ No EPUB structure files extracted!")
        
        print(f"\n🔍 Pre-flight check readiness:")
        print(f"   ✅ HTML files: {'READY' if html_files_written > 0 else 'NOT READY'}")
        print(f"   ✅ Metadata: READY")
        print(f"   ✅ Resources: READY")

# =====================================================
# UNIFIED TRANSLATION PROCESSOR
# =====================================================
class TranslationProcessor:
    """Handles the core translation processing logic"""
    
    def __init__(self, config, client, out_dir, log_callback=None, stop_callback=None):
        self.config = config
        self.client = client
        self.out_dir = out_dir
        self.log_callback = log_callback
        self.stop_callback = stop_callback
        self.chapter_splitter = ChapterSplitter(model_name=config.MODEL)
        
    def check_stop(self):
        """Check if translation should stop"""
        if self.stop_callback and self.stop_callback():
            print("❌ Translation stopped by user request.")
            return True
        return is_stop_requested()
    
    def check_duplicate_content(self, result, idx, prog, out):
        """Check if translated content is duplicate - with mode selection"""
        if not self.config.RETRY_DUPLICATE_BODIES:
            print("    ⚠️ DEBUG: Duplicate detection is DISABLED in config")
            return False, 0
        
        # Get detection mode from config
        detection_mode = getattr(self.config, 'DUPLICATE_DETECTION_MODE', 'basic')
        print(f"    🔍 DEBUG: Detection mode = '{detection_mode}'")
        print(f"    🔍 DEBUG: Lookback chapters = {self.config.DUPLICATE_LOOKBACK_CHAPTERS}")
        
        if detection_mode == 'ai-hunter':
            print("    🤖 DEBUG: Routing to AI Hunter detection...")
            return self._check_duplicate_ai_hunter(result, idx, prog, out)
        elif detection_mode == 'cascading':
            print("    🔄 DEBUG: Routing to Cascading detection...")
            return self._check_duplicate_cascading(result, idx, prog, out)
        else:
            print("    📋 DEBUG: Routing to Basic detection...")
            return self._check_duplicate_basic(result, idx, prog, out)

    def _check_duplicate_basic(self, result, idx, prog, out):
        """Original basic duplicate detection"""
        try:
            result_clean = re.sub(r'<[^>]+>', '', result).strip().lower()
            result_sample = result_clean[:1000]
            
            lookback_chapters = self.config.DUPLICATE_LOOKBACK_CHAPTERS
            
            for prev_idx in range(max(0, idx - lookback_chapters), idx):
                prev_key = str(prev_idx)
                if prev_key in prog["chapters"] and prog["chapters"][prev_key].get("output_file"):
                    prev_file = prog["chapters"][prev_key]["output_file"]
                    prev_path = os.path.join(out, prev_file)
                    
                    if os.path.exists(prev_path):
                        try:
                            with open(prev_path, 'r', encoding='utf-8') as f:
                                prev_content = f.read()
                                prev_clean = re.sub(r'<[^>]+>', '', prev_content).strip().lower()
                                prev_sample = prev_clean[:1000]
                                
                                # Use SequenceMatcher for similarity comparison
                                similarity = SequenceMatcher(None, result_sample, prev_sample).ratio()
                                
                                if similarity >= 0.85:  # 85% threshold
                                    print(f"    🚀 Basic detection: Duplicate found ({int(similarity*100)}%)")
                                    return True, int(similarity * 100)
                                    
                        except Exception as e:
                            print(f"    Warning: Failed to read {prev_path}: {e}")
                            continue
            
            return False, 0
            
        except Exception as e:
            print(f"    Warning: Failed to check duplicate content: {e}")
            return False, 0

    def check_duplicate_content(self, result, idx, prog, out):
        """Check if translated content is duplicate - with mode selection"""
        if not self.config.RETRY_DUPLICATE_BODIES:
            print("    ⚠️ DEBUG: Duplicate detection is DISABLED in config")
            return False, 0
        
        # Get detection mode from config
        detection_mode = getattr(self.config, 'DUPLICATE_DETECTION_MODE', 'basic')
        print(f"    🔍 DEBUG: Detection mode = '{detection_mode}'")
        print(f"    🔍 DEBUG: Lookback chapters = {self.config.DUPLICATE_LOOKBACK_CHAPTERS}")
        
        if detection_mode == 'ai-hunter':
            print("    🤖 DEBUG: Routing to AI Hunter detection...")
            return self._check_duplicate_ai_hunter(result, idx, prog, out)
        elif detection_mode == 'cascading':
            print("    🔄 DEBUG: Routing to Cascading detection...")
            return self._check_duplicate_cascading(result, idx, prog, out)
        else:
            print("    📋 DEBUG: Routing to Basic detection...")
            return self._check_duplicate_basic(result, idx, prog, out)

    def _check_duplicate_ai_hunter(self, result, idx, prog, out):
        """Enhanced AI Hunter duplicate detection with debug logging"""
        try:
            print(f"\n    ========== AI HUNTER DEBUG START ==========")
            print(f"    📍 Current chapter index: {idx}")
            
            # Get threshold from environment or config
            env_threshold = os.getenv('AI_HUNTER_THRESHOLD')
            config_threshold = getattr(self.config, 'AI_HUNTER_THRESHOLD', None)
            
            print(f"    🔍 Threshold sources:")
            print(f"       - Environment var: {env_threshold}")
            print(f"       - Config object: {config_threshold}")
            
            if env_threshold and env_threshold.strip():
                try:
                    threshold = float(env_threshold) / 100.0
                    threshold = max(0.1, min(1.0, threshold))
                    print(f"    ✅ Using environment threshold: {int(threshold*100)}%")
                except ValueError:
                    threshold = 0.75
                    print(f"    ⚠️ Invalid env threshold, using default: 75%")
            else:
                threshold = 0.75
                print(f"    ℹ️ No env threshold, using default: 75%")
            
            # Clean and prepare text
            result_clean = re.sub(r'<[^>]+>', '', result).strip()
            print(f"    📄 Text length: {len(result_clean)} chars")
            
            # Extract features
            print(f"    🔬 Extracting text features...")
            result_features = self._extract_text_features(result_clean)
            
            # Log extracted features
            print(f"    📊 Extracted features:")
            print(f"       - Characters found: {len(result_features.get('characters', []))}")
            print(f"       - Dialogue count: {result_features.get('semantic', {}).get('dialogue_count', 0)}")
            print(f"       - Top characters: {result_features.get('characters', [])[:5]}")
            
            lookback_chapters = self.config.DUPLICATE_LOOKBACK_CHAPTERS
            print(f"    🔄 Checking {lookback_chapters} previous chapters...")
            
            highest_similarity = 0
            detected_method = None
            all_similarities = []
            
            for prev_idx in range(max(0, idx - lookback_chapters), idx):
                prev_key = str(prev_idx)
                print(f"\n    📖 Checking against chapter {prev_idx + 1}...")
                
                if prev_key not in prog["chapters"]:
                    print(f"       ❌ Chapter not in progress data")
                    continue
                    
                if not prog["chapters"][prev_key].get("output_file"):
                    print(f"       ❌ No output file for this chapter")
                    continue
                    
                prev_file = prog["chapters"][prev_key]["output_file"]
                prev_path = os.path.join(out, prev_file)
                
                if not os.path.exists(prev_path):
                    print(f"       ❌ Output file doesn't exist: {prev_file}")
                    continue
                    
                try:
                    with open(prev_path, 'r', encoding='utf-8') as f:
                        prev_content = f.read()
                        prev_clean = re.sub(r'<[^>]+>', '', prev_content).strip()
                        
                    print(f"       ✅ Loaded {len(prev_clean)} chars from {prev_file}")
                    
                    # Extract features for previous chapter
                    prev_features = self._extract_text_features(prev_clean)
                    
                    # Multi-method detection
                    similarities = {}
                    
                    # Method 1: Exact content match
                    exact_sim = self._calculate_exact_similarity(result_clean[:2000], prev_clean[:2000])
                    similarities['exact'] = exact_sim
                    
                    # Method 2: Enhanced text similarity
                    text_sim = self._calculate_smart_similarity(result_clean, prev_clean)
                    similarities['text'] = text_sim
                    
                    # Method 3: Semantic fingerprint
                    semantic_sim = self._calculate_semantic_similarity(
                        result_features.get('semantic', {}), 
                        prev_features.get('semantic', {})
                    )
                    similarities['semantic'] = semantic_sim
                    
                    # Method 4: Structural signature
                    structural_sim = self._calculate_structural_similarity(
                        result_features.get('structural', {}), 
                        prev_features.get('structural', {})
                    )
                    similarities['structural'] = structural_sim
                    
                    # Method 5: Character analysis
                    char_sim = self._calculate_character_similarity(
                        result_features.get('characters', []), 
                        prev_features.get('characters', [])
                    )
                    similarities['character'] = char_sim
                    
                    # Method 6: Pattern analysis
                    pattern_sim = self._calculate_pattern_similarity(
                        result_features.get('patterns', {}), 
                        prev_features.get('patterns', {})
                    )
                    similarities['pattern'] = pattern_sim
                    
                    # Log all similarities
                    print(f"       📊 Similarity scores:")
                    for method, sim in similarities.items():
                        print(f"          - {method}: {int(sim*100)}%")
                    
                    # Track for debugging
                    all_similarities.append({
                        'chapter': prev_idx + 1,
                        'similarities': similarities.copy()
                    })
                    
                    # Find best match
                    for method, sim in similarities.items():
                        if sim > highest_similarity:
                            highest_similarity = sim
                            detected_method = method
                    
                    # Check if any method exceeds threshold
                    if highest_similarity >= threshold:
                        print(f"\n    🎯 DUPLICATE DETECTED!")
                        print(f"       Method: {detected_method}")
                        print(f"       Similarity: {int(highest_similarity*100)}%")
                        print(f"       Threshold: {int(threshold*100)}%")
                        print(f"       Match with: Chapter {prev_idx + 1}")
                        print(f"    ========== AI HUNTER DEBUG END ==========\n")
                        return True, int(highest_similarity * 100)
                        
                except Exception as e:
                    print(f"       ❌ Error analyzing chapter: {e}")
                    import traceback
                    print(f"       {traceback.format_exc()}")
                    continue
            
            # No duplicate found
            print(f"\n    ✅ No duplicate found")
            print(f"       Highest similarity: {int(highest_similarity*100)}% via {detected_method}")
            print(f"       Threshold: {int(threshold*100)}%")
            
            # Show top 3 closest matches
            if all_similarities:
                print(f"\n    📊 Top 3 closest matches:")
                sorted_chapters = sorted(all_similarities, 
                                       key=lambda x: max(x['similarities'].values()), 
                                       reverse=True)[:3]
                for i, chapter_data in enumerate(sorted_chapters, 1):
                    best_method = max(chapter_data['similarities'].items(), 
                                    key=lambda x: x[1])
                    print(f"       {i}. Chapter {chapter_data['chapter']}: "
                          f"{int(best_method[1]*100)}% ({best_method[0]})")
            
            print(f"    ========== AI HUNTER DEBUG END ==========\n")
            return False, int(highest_similarity * 100)
            
        except Exception as e:
            print(f"    ❌ AI Hunter detection failed with error: {e}")
            import traceback
            print(f"    {traceback.format_exc()}")
            print(f"    ========== AI HUNTER DEBUG END ==========\n")
            return False, 0


    def _check_duplicate_cascading(self, result, idx, prog, out):
        """Cascading detection - basic first, then AI Hunter for borderline cases"""
        # Step 1: Basic detection
        is_duplicate_basic, similarity_basic = self._check_duplicate_basic(result, idx, prog, out)
        
        if is_duplicate_basic:
            return True, similarity_basic
        
        # Step 2: If basic detection finds moderate similarity, use AI Hunter
        if similarity_basic >= 60:  # Configurable threshold
            print(f"    🤖 Moderate similarity ({similarity_basic}%) - running AI Hunter analysis...")
            is_duplicate_ai, similarity_ai = self._check_duplicate_ai_hunter(result, idx, prog, out)
            
            if is_duplicate_ai:
                return True, similarity_ai
        
        return False, max(similarity_basic, 0)

    def _extract_text_features(self, text):
        """Extract multiple features from text for AI Hunter analysis"""
        features = {
            'semantic': {},
            'structural': {},
            'characters': [],
            'patterns': {}
        }
        
        # Semantic fingerprint
        lines = text.split('\n')
        
        # Character extraction (names that appear 3+ times)
        words = re.findall(r'\b[A-Z][a-z]+\b', text)
        word_freq = Counter(words)
        features['characters'] = [name for name, count in word_freq.items() if count >= 3]
        
        # Dialogue patterns
        dialogue_patterns = re.findall(r'"([^"]+)"', text)
        features['semantic']['dialogue_count'] = len(dialogue_patterns)
        features['semantic']['dialogue_lengths'] = [len(d) for d in dialogue_patterns[:10]]
        
        # Speaker patterns
        speaker_patterns = re.findall(r'(\w+)\s+(?:said|asked|replied|shouted|whispered)', text.lower())
        features['semantic']['speakers'] = list(set(speaker_patterns[:20]))
        
        # Number extraction
        numbers = re.findall(r'\b\d+\b', text)
        features['patterns']['numbers'] = numbers[:20]
        
        # Structural signature
        para_lengths = []
        dialogue_count = 0
        for para in text.split('\n\n'):
            if para.strip():
                para_lengths.append(len(para))
                if '"' in para:
                    dialogue_count += 1
        
        features['structural']['para_count'] = len(para_lengths)
        features['structural']['avg_para_length'] = sum(para_lengths) / max(1, len(para_lengths))
        features['structural']['dialogue_ratio'] = dialogue_count / max(1, len(para_lengths))
        
        # Create structural pattern string
        pattern = []
        for para in text.split('\n\n')[:20]:  # First 20 paragraphs
            if para.strip():
                if '"' in para:
                    pattern.append('D')  # Dialogue
                elif len(para) > 300:
                    pattern.append('L')  # Long
                elif len(para) < 100:
                    pattern.append('S')  # Short
                else:
                    pattern.append('M')  # Medium
        features['structural']['pattern'] = ''.join(pattern)
        
        return features

    def _calculate_exact_similarity(self, text1, text2):
        """Calculate exact text similarity"""
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

    def _calculate_smart_similarity(self, text1, text2):
        """Smart similarity with length-aware sampling"""
        # Check length ratio first
        len_ratio = len(text1) / max(1, len(text2))
        if len_ratio < 0.7 or len_ratio > 1.3:
            return 0.0
        
        # Smart sampling for large texts
        if len(text1) > 10000:
            sample_size = 3000
            samples1 = [
                text1[:sample_size],
                text1[len(text1)//2 - sample_size//2:len(text1)//2 + sample_size//2],
                text1[-sample_size:]
            ]
            samples2 = [
                text2[:sample_size],
                text2[len(text2)//2 - sample_size//2:len(text2)//2 + sample_size//2],
                text2[-sample_size:]
            ]
            similarities = [SequenceMatcher(None, s1.lower(), s2.lower()).ratio() 
                           for s1, s2 in zip(samples1, samples2)]
            return sum(similarities) / len(similarities)
        else:
            # Use first 2000 chars for smaller texts
            return SequenceMatcher(None, text1[:2000].lower(), text2[:2000].lower()).ratio()

    def _calculate_semantic_similarity(self, sem1, sem2):
        """Calculate semantic fingerprint similarity"""
        score = 0.0
        max_score = 0.0
        
        # Compare dialogue counts
        if 'dialogue_count' in sem1 and 'dialogue_count' in sem2:
            max_score += 1.0
            ratio = min(sem1['dialogue_count'], sem2['dialogue_count']) / max(1, max(sem1['dialogue_count'], sem2['dialogue_count']))
            score += ratio * 0.3
        
        # Compare speakers
        if 'speakers' in sem1 and 'speakers' in sem2:
            max_score += 1.0
            if sem1['speakers'] and sem2['speakers']:
                overlap = len(set(sem1['speakers']) & set(sem2['speakers']))
                total = len(set(sem1['speakers']) | set(sem2['speakers']))
                score += (overlap / max(1, total)) * 0.4
        
        # Compare dialogue lengths pattern
        if 'dialogue_lengths' in sem1 and 'dialogue_lengths' in sem2:
            max_score += 1.0
            if sem1['dialogue_lengths'] and sem2['dialogue_lengths']:
                # Compare dialogue length patterns
                len1 = sem1['dialogue_lengths'][:10]
                len2 = sem2['dialogue_lengths'][:10]
                if len1 and len2:
                    avg1 = sum(len1) / len(len1)
                    avg2 = sum(len2) / len(len2)
                    ratio = min(avg1, avg2) / max(1, max(avg1, avg2))
                    score += ratio * 0.3
        
        return score / max(1, max_score)

    def _calculate_structural_similarity(self, struct1, struct2):
        """Calculate structural signature similarity"""
        score = 0.0
        
        # Compare paragraph patterns
        if 'pattern' in struct1 and 'pattern' in struct2:
            pattern_sim = SequenceMatcher(None, struct1['pattern'], struct2['pattern']).ratio()
            score += pattern_sim * 0.4
        
        # Compare paragraph statistics
        if all(k in struct1 for k in ['para_count', 'avg_para_length', 'dialogue_ratio']) and \
           all(k in struct2 for k in ['para_count', 'avg_para_length', 'dialogue_ratio']):
            
            # Paragraph count ratio
            para_ratio = min(struct1['para_count'], struct2['para_count']) / max(1, max(struct1['para_count'], struct2['para_count']))
            score += para_ratio * 0.2
            
            # Average length ratio
            avg_ratio = min(struct1['avg_para_length'], struct2['avg_para_length']) / max(1, max(struct1['avg_para_length'], struct2['avg_para_length']))
            score += avg_ratio * 0.2
            
            # Dialogue ratio similarity
            dialogue_diff = abs(struct1['dialogue_ratio'] - struct2['dialogue_ratio'])
            score += (1 - dialogue_diff) * 0.2
        
        return score

    def _calculate_character_similarity(self, chars1, chars2):
        """Calculate character name similarity"""
        if not chars1 or not chars2:
            return 0.0
        
        # Find overlapping characters
        set1 = set(chars1)
        set2 = set(chars2)
        overlap = len(set1 & set2)
        total = len(set1 | set2)
        
        return overlap / max(1, total)

    def _calculate_pattern_similarity(self, pat1, pat2):
        """Calculate pattern-based similarity"""
        score = 0.0
        
        # Compare numbers (they rarely change in translations)
        if 'numbers' in pat1 and 'numbers' in pat2:
            nums1 = set(pat1['numbers'])
            nums2 = set(pat2['numbers'])
            if nums1 and nums2:
                overlap = len(nums1 & nums2)
                total = len(nums1 | nums2)
                score = overlap / max(1, total)
        
        return score
    
    def generate_rolling_summary(self, history_manager, idx, chunk_idx):
        """Generate rolling summary for context continuity"""
        if not self.config.USE_ROLLING_SUMMARY or not self.config.CONTEXTUAL:
            return None
            
        current_history = history_manager.load_history()
        
        messages_to_include = self.config.ROLLING_SUMMARY_EXCHANGES * 2
        
        if len(current_history) >= 4:
            assistant_responses = []
            recent_messages = current_history[-messages_to_include:] if messages_to_include > 0 else current_history
            
            for h in recent_messages:
                if h.get("role") == "assistant":
                    assistant_responses.append(h["content"])
            
            if assistant_responses:
                system_prompt = os.getenv("ROLLING_SUMMARY_SYSTEM_PROMPT", 
                                        "Create a concise summary for context continuity.")
                user_prompt_template = os.getenv("ROLLING_SUMMARY_USER_PROMPT",
                                                "Summarize the key events, characters, tone, and important details from these translations. "
                                                "Focus on: character names/relationships, plot developments, and any special terminology used.\n\n"
                                                "{translations}")
                
                translations_text = "\n---\n".join(assistant_responses)
                user_prompt = user_prompt_template.replace("{translations}", translations_text)
                
                summary_msgs = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
                
                try:
                    summary_resp, _ = send_with_interrupt(
                        summary_msgs, self.client, self.config.TEMP, 
                        min(2000, self.config.MAX_OUTPUT_TOKENS), 
                        self.check_stop
                    )
                    
                    summary_file = os.path.join(self.out_dir, "rolling_summary.txt")
                    
                    if self.config.ROLLING_SUMMARY_MODE == "append":
                        mode = "a"
                        with open(summary_file, mode, encoding="utf-8") as sf:
                            sf.write(f"\n\n=== Summary before chapter {idx+1}, chunk {chunk_idx} ===\n")
                            sf.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}]\n")
                            sf.write(summary_resp.strip())
                    else:
                        mode = "w"
                        with open(summary_file, mode, encoding="utf-8") as sf:
                            sf.write(f"=== Latest Summary (Chapter {idx+1}, chunk {chunk_idx}) ===\n")
                            sf.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}]\n")
                            sf.write(summary_resp.strip())
                    
                    print(f"📝 Generated rolling summary ({self.config.ROLLING_SUMMARY_MODE} mode, {self.config.ROLLING_SUMMARY_EXCHANGES} exchanges)")
                    
                    if self.config.ROLLING_SUMMARY_MODE == "append" and os.path.exists(summary_file):
                        try:
                            with open(summary_file, 'r', encoding='utf-8') as sf:
                                content = sf.read()
                            summary_count = content.count("=== Summary")
                            print(f"   📚 Total summaries in memory: {summary_count}")
                        except:
                            pass
                    
                    return summary_resp.strip()
                    
                except Exception as e:
                    print(f"⚠️ Failed to generate rolling summary: {e}")
                    return None
        
        return None
    
    def translate_with_retry(self, msgs, chunk_html, c, chunk_idx, total_chunks):
        """Handle translation with retry logic"""
        retry_count = 0
        max_retries = 3
        duplicate_retry_count = 0
        max_duplicate_retries = 6
        timeout_retry_count = 0
        max_timeout_retries = 2
        history_purged = False
        
        original_max_tokens = self.config.MAX_OUTPUT_TOKENS
        original_temp = self.config.TEMP
        original_user_prompt = msgs[-1]["content"]
        
        chunk_timeout = None
        if self.config.RETRY_TIMEOUT:
            chunk_timeout = self.config.CHUNK_TIMEOUT
        
        result = None
        finish_reason = None
        
        while True:
            if self.check_stop():
                return None, None
            
            try:
                current_max_tokens = self.config.MAX_OUTPUT_TOKENS
                current_temp = self.config.TEMP
                
                total_tokens = sum(self.chapter_splitter.count_tokens(m["content"]) for m in msgs)
                print(f"[DEBUG] Chunk {chunk_idx}/{total_chunks} tokens = {total_tokens:,} / {self.get_token_budget_str()} [File: {c['original_basename'] if 'original_basename' in c else 'Chapter_'+str(c['num'])}]")
                
                self.client.context = 'translation'
                
                result, finish_reason = send_with_interrupt(
                    msgs, self.client, current_temp, current_max_tokens, 
                    self.check_stop, chunk_timeout
                )
                
                retry_needed = False
                retry_reason = ""
                is_duplicate_retry = False
                
                if finish_reason == "length" and self.config.RETRY_TRUNCATED:
                    if retry_count < max_retries:
                        retry_needed = True
                        retry_reason = "truncated output"
                        self.config.MAX_OUTPUT_TOKENS = min(self.config.MAX_OUTPUT_TOKENS * 2, self.config.MAX_RETRY_TOKENS)
                        retry_count += 1
                
                if not retry_needed and self.config.RETRY_DUPLICATE_BODIES:
                    if duplicate_retry_count < max_duplicate_retries:
                        idx = c.get('__index', 0)
                        prog = c.get('__progress', {})
                        is_duplicate, similarity = self.check_duplicate_content(result, idx, prog, self.out_dir)
                        
                        if is_duplicate:
                            retry_needed = True
                            is_duplicate_retry = True
                            retry_reason = f"duplicate content (similarity: {similarity}%)"
                            duplicate_retry_count += 1
                            
                            if duplicate_retry_count >= 3 and not history_purged:
                                print(f"    🧹 Clearing history after 3 attempts...")
                                if 'history_manager' in c:
                                    c['history_manager'].save_history([])
                                history_purged = True
                                self.config.TEMP = original_temp
                            
                            elif duplicate_retry_count == 1:
                                print(f"    🔄 First duplicate retry - same temperature")
                            
                            elif history_purged:
                                attempts_since_purge = duplicate_retry_count - 3
                                self.config.TEMP = min(original_temp + (0.1 * attempts_since_purge), 1.0)
                                print(f"    🌡️ Post-purge temp: {self.config.TEMP}")
                            
                            else:
                                self.config.TEMP = min(original_temp + (0.1 * (duplicate_retry_count - 1)), 1.0)
                                print(f"    🌡️ Gradual temp increase: {self.config.TEMP}")
                            
                            if duplicate_retry_count == 1:
                                user_prompt = f"[RETRY] Chapter {c['num']}: Ensure unique translation.\n{chunk_html}"
                            elif duplicate_retry_count <= 3:
                                user_prompt = f"[ATTEMPT {duplicate_retry_count}] Translate uniquely:\n{chunk_html}"
                            else:
                                user_prompt = f"Chapter {c['num']}:\n{chunk_html}"
                            
                            msgs[-1] = {"role": "user", "content": user_prompt}
                
                if retry_needed:
                    if is_duplicate_retry:
                        print(f"    🔄 Duplicate retry {duplicate_retry_count}/{max_duplicate_retries}")
                    else:
                        print(f"    🔄 Retry {retry_count}/{max_retries}: {retry_reason}")
                    
                    time.sleep(2)
                    continue
                
                break
                
            except UnifiedClientError as e:
                error_msg = str(e)
                
                if "stopped by user" in error_msg:
                    print("❌ Translation stopped by user during API call")
                    return None, None
                
                if "took" in error_msg and "timeout:" in error_msg:
                    if timeout_retry_count < max_timeout_retries:
                        timeout_retry_count += 1
                        print(f"    ⏱️ Chunk took too long, retry {timeout_retry_count}/{max_timeout_retries}")
                        print(f"    🔄 Retrying")
                        time.sleep(2)
                        continue
                    else:
                        print(f"    ❌ Max timeout retries reached")
                        raise UnifiedClientError("Translation failed after timeout retries")
                
                elif "timed out" in error_msg and "timeout:" not in error_msg:
                    print(f"⚠️ {error_msg}, retrying...")
                    time.sleep(5)
                    continue
                
                elif getattr(e, "http_status", None) == 429:
                    print("⚠️ Rate limited, sleeping 60s…")
                    for i in range(60):
                        if self.check_stop():
                            print("❌ Translation stopped during rate limit wait")
                            return None, None
                        time.sleep(1)
                    continue
                
                else:
                    raise
            
            except Exception as e:
                print(f"❌ Unexpected error during API call: {e}")
                raise
        
        self.config.MAX_OUTPUT_TOKENS = original_max_tokens
        self.config.TEMP = original_temp
        
        if retry_count > 0 or duplicate_retry_count > 0 or timeout_retry_count > 0:
            if duplicate_retry_count > 0:
                print(f"    🔄 Restored original temperature: {self.config.TEMP} (after {duplicate_retry_count} duplicate retries)")
            elif timeout_retry_count > 0:
                print(f"    🔄 Restored original settings after {timeout_retry_count} timeout retries")
            elif retry_count > 0:
                print(f"    🔄 Restored original settings after {retry_count} retries")
        
        if duplicate_retry_count >= max_duplicate_retries:
            print(f"    ⚠️ WARNING: Duplicate content issue persists after {max_duplicate_retries} attempts")
        
        return result, finish_reason
    
    def get_token_budget_str(self):
        """Get token budget as string"""
        _tok_env = os.getenv("MAX_INPUT_TOKENS", "1000000").strip()
        max_tokens_limit, budget_str = parse_token_limit(_tok_env)
        return budget_str

# =====================================================
# BATCH TRANSLATION PROCESSOR
# =====================================================
class BatchTranslationProcessor:
    """Handles batch/parallel translation processing"""
    
    def __init__(self, config, client, base_msg, out_dir, progress_lock, 
                 save_progress_fn, update_progress_fn, check_stop_fn, 
                 image_translator=None):
        self.config = config
        self.client = client
        self.base_msg = base_msg
        self.out_dir = out_dir
        self.progress_lock = progress_lock
        self.save_progress_fn = save_progress_fn
        self.update_progress_fn = update_progress_fn
        self.check_stop_fn = check_stop_fn
        self.image_translator = image_translator
        self.chapters_completed = 0
        self.chunks_completed = 0
    
    def process_single_chapter(self, chapter_data):
        """Process a single chapter (runs in thread)"""
        idx, chapter = chapter_data
        chap_num = chapter["num"]
        
        actual_num = FileUtilities.extract_actual_chapter_number(chapter)
        
        try:
            print(f"🔄 Starting #{idx+1} (Internal: Chapter {chap_num}, Actual: Chapter {actual_num})  (thread: {threading.current_thread().name}) [File: {chapter.get('original_basename', f'Chapter_{chap_num}')}]")
            
            content_hash = chapter.get("content_hash") or ContentProcessor.get_content_hash(chapter["body"])
            with self.progress_lock:
                self.update_progress_fn(idx, actual_num, content_hash, None, status="in_progress")
                self.save_progress_fn()
            
            chapter_body = chapter["body"]
            if chapter.get('has_images') and self.image_translator and self.config.ENABLE_IMAGE_TRANSLATION:
                print(f"🖼️ Processing images for Chapter {actual_num}...")
                self.image_translator.set_current_chapter(actual_num)
                chapter_body, image_translations = process_chapter_images(
                    chapter_body, 
                    actual_num, 
                    self.image_translator,
                    self.check_stop_fn
                )
                if image_translations:
                    print(f"✅ Processed {len(image_translations)} images for Chapter {actual_num}")
            
            chapter_msgs = self.base_msg + [{"role": "user", "content": chapter_body}]
            
            print(f"📤 Sending Chapter {actual_num} to API...")
            result, finish_reason = send_with_interrupt(
                chapter_msgs, self.client, self.config.TEMP, 
                self.config.MAX_OUTPUT_TOKENS, self.check_stop_fn
            )
            
            print(f"📥 Received Chapter {actual_num} response, finish_reason: {finish_reason}")
            
            if finish_reason in ["length", "max_tokens"]:
                print(f"⚠️ Chapter {actual_num} response was TRUNCATED!")
            
            if self.config.REMOVE_AI_ARTIFACTS:
                result = ContentProcessor.clean_ai_artifacts(result, True)
                
            result = ContentProcessor.clean_memory_artifacts(result)
            
            if self.config.EMERGENCY_RESTORE:
                result = ContentProcessor.emergency_restore_paragraphs(result, chapter_body)
            
            cleaned = re.sub(r"^```(?:html)?\s*\n?", "", result, count=1, flags=re.MULTILINE)
            cleaned = re.sub(r"\n?```\s*$", "", cleaned, count=1, flags=re.MULTILINE)
            cleaned = ContentProcessor.clean_ai_artifacts(cleaned, remove_artifacts=self.config.REMOVE_AI_ARTIFACTS)
            
            fname = FileUtilities.create_chapter_filename(chapter, actual_num)
            
            with open(os.path.join(self.out_dir, fname), 'w', encoding='utf-8') as f:
                f.write(cleaned)
            
            print(f"💾 Saved Chapter {actual_num}: {fname} ({len(cleaned)} chars)")
            
            with self.progress_lock:
                self.update_progress_fn(idx, actual_num, content_hash, fname, status="completed")
                self.save_progress_fn()
                
                self.chapters_completed += 1
                self.chunks_completed += 1
            
            print(f"✅ Chapter {actual_num} completed successfully")
            return True, actual_num
            
        except Exception as e:
            print(f"❌ Chapter {actual_num} failed: {e}")
            with self.progress_lock:
                self.update_progress_fn(idx, actual_num, content_hash, None, status="failed")
                self.save_progress_fn()
            return False, actual_num

# =====================================================
# GLOSSARY MANAGER
# =====================================================
class GlossaryManager:
    """Unified glossary management"""
    
    def __init__(self):
        self.pattern_manager = PatternManager()
    
    def save_glossary(self, output_dir, chapters, instructions, language="korean"):
        """Targeted glossary generator - Focuses on titles and CJK names with honorifics"""
        print("📑 Targeted Glossary Generator v4.0 (Enhanced)")
        print("📑 Extracting complete names with honorifics and titles")
        
        manual_glossary_path = os.getenv("MANUAL_GLOSSARY")
        if manual_glossary_path and os.path.exists(manual_glossary_path):
            print(f"📑 Manual glossary detected: {os.path.basename(manual_glossary_path)}")
            
            target_path = os.path.join(output_dir, "glossary.json")
            try:
                shutil.copy2(manual_glossary_path, target_path)
                print(f"📑 ✅ Manual glossary copied to: {target_path}")
                print(f"📑 Skipping automatic glossary generation")
                
                with open(manual_glossary_path, 'r', encoding='utf-8') as f:
                    manual_data = json.load(f)
                
                simple_entries = {}
                if isinstance(manual_data, list):
                    for char in manual_data:
                        original = char.get('original_name', '')
                        translated = char.get('name', original)
                        if original and translated:
                            simple_entries[original] = translated
                
                simple_path = os.path.join(output_dir, "glossary_simple.json")
                with open(simple_path, 'w', encoding='utf-8') as f:
                    json.dump(simple_entries, f, ensure_ascii=False, indent=2)
                
                return simple_entries
                
            except Exception as e:
                print(f"⚠️ Could not copy manual glossary: {e}")
                print(f"📑 Proceeding with automatic generation...")
        
        glossary_folder_path = os.path.join(output_dir, "Glossary")
        existing_glossary = None
        
        if os.path.exists(glossary_folder_path):
            for file in os.listdir(glossary_folder_path):
                if file.endswith("_glossary.json"):
                    existing_path = os.path.join(glossary_folder_path, file)
                    try:
                        with open(existing_path, 'r', encoding='utf-8') as f:
                            existing_glossary = json.load(f)
                        print(f"📑 Found existing glossary from manual extraction: {file}")
                        break
                    except Exception as e:
                        print(f"⚠️ Could not load existing glossary: {e}")
        
        min_frequency = int(os.getenv("GLOSSARY_MIN_FREQUENCY", "2"))
        max_names = int(os.getenv("GLOSSARY_MAX_NAMES", "50"))
        max_titles = int(os.getenv("GLOSSARY_MAX_TITLES", "30"))
        batch_size = int(os.getenv("GLOSSARY_BATCH_SIZE", "50"))
        
        print(f"📑 Settings: Min frequency: {min_frequency}, Max names: {max_names}, Max titles: {max_titles}")
        
        custom_prompt = os.getenv("AUTO_GLOSSARY_PROMPT", "").strip()
        
        def clean_html(html_text):
            """Remove HTML tags to get clean text"""
            soup = BeautifulSoup(html_text, 'html.parser')
            return soup.get_text()
        
        all_text = ' '.join(clean_html(chapter["body"]) for chapter in chapters)
        print(f"📑 Processing {len(all_text):,} characters of text")
        
        if custom_prompt:
            return self._extract_with_custom_prompt(custom_prompt, all_text, language, 
                                                   min_frequency, max_names, max_titles, 
                                                   existing_glossary, output_dir)
        else:
            return self._extract_with_patterns(all_text, language, min_frequency, 
                                             max_names, max_titles, batch_size, 
                                             existing_glossary, output_dir)
    
    def _extract_with_custom_prompt(self, custom_prompt, all_text, language, 
                                   min_frequency, max_names, max_titles, 
                                   existing_glossary, output_dir):
        """Extract glossary using custom AI prompt"""
        print("📑 Using custom automatic glossary prompt")
        
        try:
            MODEL = os.getenv("MODEL", "gemini-1.5-flash")
            API_KEY = (os.getenv("API_KEY") or 
                       os.getenv("OPENAI_API_KEY") or 
                       os.getenv("OPENAI_OR_Gemini_API_KEY") or
                       os.getenv("GEMINI_API_KEY"))
            
            if not API_KEY:
                print(f"📑 No API key found, falling back to pattern-based extraction")
                return self._extract_with_patterns(all_text, language, min_frequency, 
                                                 max_names, max_titles, 50, 
                                                 existing_glossary, output_dir)
            else:
                print(f"📑 Using AI-assisted extraction with custom prompt")
                
                from unified_api_client import UnifiedClient
                client = UnifiedClient(model=MODEL, api_key=API_KEY)
                
                text_sample = all_text[:50000] if len(all_text) > 50000 else all_text
                
                prompt = custom_prompt.replace('{language}', language)
                prompt = prompt.replace('{min_frequency}', str(min_frequency))
                prompt = prompt.replace('{max_names}', str(max_names))
                prompt = prompt.replace('{max_titles}', str(max_titles))
                
                messages = [
                    {"role": "system", "content": "You are a glossary extraction assistant. Extract names and terms as requested."},
                    {"role": "user", "content": f"{prompt}\n\nText sample:\n{text_sample}"}
                ]
                
                try:
                    response, _ = client.send(messages, temperature=0.3, max_tokens=4096)
                    
                    ai_extracted_terms = {}
                    
                    try:
                        ai_data = json.loads(response)
                        if isinstance(ai_data, dict):
                            ai_extracted_terms = ai_data
                        elif isinstance(ai_data, list):
                            for item in ai_data:
                                if isinstance(item, dict) and 'original' in item:
                                    ai_extracted_terms[item['original']] = item.get('translation', item['original'])
                    except:
                        lines = response.strip().split('\n')
                        for line in lines:
                            if '->' in line or '→' in line or ':' in line:
                                parts = re.split(r'->|→|:', line, 1)
                                if len(parts) == 2:
                                    original = parts[0].strip()
                                    translation = parts[1].strip()
                                    if original and translation:
                                        ai_extracted_terms[original] = translation
                    
                    print(f"📑 AI extracted {len(ai_extracted_terms)} initial terms")
                    
                    verified_terms = {}
                    for original, translation in ai_extracted_terms.items():
                        count = all_text.count(original)
                        if count >= min_frequency:
                            verified_terms[original] = translation
                    
                    print(f"📑 Verified {len(verified_terms)} terms meet frequency threshold")
                    
                    if existing_glossary and isinstance(existing_glossary, list):
                        print("📑 Merging with existing manual glossary...")
                        merged_count = 0
                        
                        for char in existing_glossary:
                            original = char.get('original_name', '')
                            translated = char.get('name', original)
                            
                            if original and translated and original not in verified_terms:
                                verified_terms[original] = translated
                                merged_count += 1
                        
                        print(f"📑 Added {merged_count} entries from existing glossary")
                    
                    glossary_data = {
                        "metadata": {
                            "language": language,
                            "extraction_method": "ai_custom_prompt",
                            "total_entries": len(verified_terms),
                            "min_frequency": min_frequency,
                            "generation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "merged_with_manual": existing_glossary is not None
                        },
                        "entries": verified_terms
                    }
                    
                    glossary_path = os.path.join(output_dir, "glossary.json")
                    with open(glossary_path, 'w', encoding='utf-8') as f:
                        json.dump(glossary_data, f, ensure_ascii=False, indent=2)
                    
                    print(f"\n📑 ✅ AI-ASSISTED GLOSSARY SAVED!")
                    print(f"📑 File: {glossary_path}")
                    print(f"📑 Total entries: {len(verified_terms)}")
                    
                    simple_glossary_path = os.path.join(output_dir, "glossary_simple.json")
                    with open(simple_glossary_path, 'w', encoding='utf-8') as f:
                        json.dump(verified_terms, f, ensure_ascii=False, indent=2)
                    
                    return verified_terms
                    
                except Exception as e:
                    print(f"⚠️ AI extraction failed: {e}")
                    print("📑 Falling back to pattern-based extraction")
                    return self._extract_with_patterns(all_text, language, min_frequency, 
                                                     max_names, max_titles, 50, 
                                                     existing_glossary, output_dir)
                    
        except Exception as e:
            print(f"⚠️ Custom prompt processing failed: {e}")
            return self._extract_with_patterns(all_text, language, min_frequency, 
                                             max_names, max_titles, 50, 
                                             existing_glossary, output_dir)
    
    def _extract_with_patterns(self, all_text, language, min_frequency, 
                               max_names, max_titles, batch_size, 
                               existing_glossary, output_dir):
        """Extract glossary using pattern matching"""
        print("📑 Using pattern-based extraction")
        
        def is_valid_name(name, language_hint='unknown'):
            """Strict validation for proper names only"""
            if not name or len(name.strip()) < 1:
                return False
                
            name = name.strip()
            
            if name.lower() in self.pattern_manager.COMMON_WORDS or name in self.pattern_manager.COMMON_WORDS:
                return False
            
            if language_hint == 'korean':
                if not (2 <= len(name) <= 4):
                    return False
                if not all(0xAC00 <= ord(char) <= 0xD7AF for char in name):
                    return False
                if len(set(name)) == 1:
                    return False
                    
            elif language_hint == 'japanese':
                if not (2 <= len(name) <= 6):
                    return False
                has_kanji = any(0x4E00 <= ord(char) <= 0x9FFF for char in name)
                has_kana = any((0x3040 <= ord(char) <= 0x309F) or (0x30A0 <= ord(char) <= 0x30FF) for char in name)
                if not (has_kanji or has_kana):
                    return False
                    
            elif language_hint == 'chinese':
                if not (2 <= len(name) <= 4):
                    return False
                if not all(0x4E00 <= ord(char) <= 0x9FFF for char in name):
                    return False
                    
            elif language_hint == 'english':
                if not name[0].isupper():
                    return False
                if sum(1 for c in name if c.isalpha()) < len(name) * 0.8:
                    return False
                if not (2 <= len(name) <= 20):
                    return False
            
            return True
        
        def detect_language_hint(text_sample):
            """Quick language detection for validation purposes with enhanced debugging"""
            # Only analyze first 1000 characters for performance
            sample = text_sample[:1000]
            sample_length = len(sample)
            
            # Count characters by script type
            korean_chars = sum(1 for char in sample if 0xAC00 <= ord(char) <= 0xD7AF)
            japanese_kana = sum(1 for char in sample if (0x3040 <= ord(char) <= 0x309F) or (0x30A0 <= ord(char) <= 0x30FF))
            chinese_chars = sum(1 for char in sample if 0x4E00 <= ord(char) <= 0x9FFF)
            latin_chars = sum(1 for char in sample if 0x0041 <= ord(char) <= 0x007A)
            
            # Calculate total CJK characters
            total_cjk = korean_chars + japanese_kana + chinese_chars
            
            # Calculate percentages
            korean_pct = (korean_chars / sample_length * 100) if sample_length > 0 else 0
            japanese_pct = (japanese_kana / sample_length * 100) if sample_length > 0 else 0
            chinese_pct = (chinese_chars / sample_length * 100) if sample_length > 0 else 0
            latin_pct = (latin_chars / sample_length * 100) if sample_length > 0 else 0
            
            # Print detailed debugging information
            print(f"\n🔍 LANGUAGE DETECTION DEBUG:")
            print(f"   📊 Sample analyzed: {sample_length} characters")
            print(f"   📈 Character counts:")
            print(f"      • Korean (Hangul): {korean_chars} chars ({korean_pct:.1f}%)")
            print(f"      • Japanese (Kana): {japanese_kana} chars ({japanese_pct:.1f}%)")
            print(f"      • Chinese (Hanzi): {chinese_chars} chars ({chinese_pct:.1f}%)")
            print(f"      • Latin/English:   {latin_chars} chars ({latin_pct:.1f}%)")
            print(f"      • Total CJK:       {total_cjk} chars ({(total_cjk/sample_length*100):.1f}%)")
            
            # Show decision thresholds
            print(f"   ⚖️ Decision thresholds:")
            print(f"      • Korean threshold: 50 chars (current: {korean_chars})")
            print(f"      • Japanese threshold: 20 chars (current: {japanese_kana})")
            print(f"      • Chinese threshold: 50 chars + Japanese < 10 (current: {chinese_chars}, JP: {japanese_kana})")
            print(f"      • English threshold: 100 chars (current: {latin_chars})")
            
            # Apply detection logic with detailed reasoning
            if korean_chars > 50:
                print(f"   ✅ DETECTED: Korean (Korean chars {korean_chars} > 50)")
                result = 'korean'
            elif japanese_kana > 20:
                print(f"   ✅ DETECTED: Japanese (Japanese kana {japanese_kana} > 20)")
                result = 'japanese'
            elif chinese_chars > 50 and japanese_kana < 10:
                print(f"   ✅ DETECTED: Chinese (Chinese chars {chinese_chars} > 50 AND Japanese kana {japanese_kana} < 10)")
                result = 'chinese'
            elif latin_chars > 100:
                print(f"   ✅ DETECTED: English (Latin chars {latin_chars} > 100)")
                result = 'english'
            else:
                print(f"   ❓ DETECTED: Unknown (no thresholds met)")
                result = 'unknown'
            
            # Show first few characters as examples
            if sample:
                # Get first 20 characters for preview
                preview = sample[:20].replace('\n', '\\n').replace('\r', '\\r')
                print(f"   📝 Sample text preview: '{preview}{'...' if len(sample) > 20 else ''}'")
                
                # Show some example characters by type
                korean_examples = [c for c in sample[:50] if 0xAC00 <= ord(c) <= 0xD7AF][:5]
                japanese_examples = [c for c in sample[:50] if (0x3040 <= ord(c) <= 0x309F) or (0x30A0 <= ord(c) <= 0x30FF)][:5]
                chinese_examples = [c for c in sample[:50] if 0x4E00 <= ord(c) <= 0x9FFF][:5]
                
                if korean_examples:
                    print(f"   🇰🇷 Korean examples: {''.join(korean_examples)}")
                if japanese_examples:
                    print(f"   🇯🇵 Japanese examples: {''.join(japanese_examples)}")
                if chinese_examples:
                    print(f"   🇨🇳 Chinese examples: {''.join(chinese_examples)}")
            
            print(f"   🎯 FINAL RESULT: {result.upper()}")
            
            return result
        
        language_hint = detect_language_hint(all_text)
        print(f"📑 Detected primary language: {language_hint}")
        
        honorifics_to_use = []
        if language_hint in self.pattern_manager.CJK_HONORIFICS:
            honorifics_to_use.extend(self.pattern_manager.CJK_HONORIFICS[language_hint])
        honorifics_to_use.extend(self.pattern_manager.CJK_HONORIFICS['english'])
        
        print(f"📑 Using {len(honorifics_to_use)} honorifics for {language_hint}")
        
        names_with_honorifics = {}
        standalone_names = {}
        
        print("📑 Scanning for names with honorifics...")
        
        # Extract names with honorifics
        for honorific in honorifics_to_use:
            self._extract_names_for_honorific(honorific, all_text, language_hint, 
                                            min_frequency, names_with_honorifics, 
                                            standalone_names, is_valid_name)
        
        # Also look for titles with honorifics
        if language_hint == 'korean':
            korean_titles = [
                '사장', '회장', '부장', '과장', '대리', '팀장', '실장', '본부장',
                '선생', '교수', '박사', '의사', '변호사', '검사', '판사',
                '대통령', '총리', '장관', '시장', '지사', '의원',
                '왕', '여왕', '왕자', '공주', '황제', '황후',
                '장군', '대장', '원수', '제독', '함장',
                '어머니', '아버지', '할머니', '할아버지',
                '사모', '선배', '후배', '동료'
            ]
            
            for title in korean_titles:
                for honorific in ['님']:
                    full_title = title + honorific
                    count = len(re.findall(re.escape(full_title) + r'(?=\s|[,.\!?]|$)', all_text))
                    
                    if count >= min_frequency:
                        names_with_honorifics[full_title] = count
        
        print(f"📑 Found {len(standalone_names)} unique names with honorifics")
        print(f"[DEBUG] Sample text (first 500 chars): {all_text[:500]}")
        print(f"[DEBUG] Text length: {len(all_text)}")
        print(f"[DEBUG] Honorifics being used: {honorifics_to_use[:5]}")
        
        # Find titles
        print("📑 Scanning for titles...")
        found_titles = {}
        
        title_patterns_to_use = []
        if language_hint in self.pattern_manager.TITLE_PATTERNS:
            title_patterns_to_use.extend(self.pattern_manager.TITLE_PATTERNS[language_hint])
        title_patterns_to_use.extend(self.pattern_manager.TITLE_PATTERNS['english'])
        
        for pattern in title_patterns_to_use:
            matches = re.finditer(pattern, all_text, re.IGNORECASE if 'english' in pattern else 0)
            
            for match in matches:
                title = match.group(0)
                
                if title in names_with_honorifics:
                    continue
                    
                count = len(re.findall(re.escape(title) + r'(?=\s|[,.\!?、。，]|$)', all_text, re.IGNORECASE if 'english' in pattern else 0))
                
                if count >= min_frequency:
                    if re.match(r'[A-Za-z]', title):
                        title = title.title()
                    
                    if title not in found_titles:
                        found_titles[title] = count
        
        print(f"📑 Found {len(found_titles)} unique titles")
        
        sorted_names = sorted(names_with_honorifics.items(), key=lambda x: x[1], reverse=True)[:max_names]
        sorted_titles = sorted(found_titles.items(), key=lambda x: x[1], reverse=True)[:max_titles]
        
        all_terms = []
        
        for name, count in sorted_names:
            all_terms.append(name)
            
        for title, count in sorted_titles:
            all_terms.append(title)
        
        print(f"📑 Total terms to translate: {len(all_terms)}")
        
        if sorted_names:
            print("\n📑 Sample names found:")
            for name, count in sorted_names[:5]:
                print(f"   • {name} ({count}x)")
        
        if sorted_titles:
            print("\n📑 Sample titles found:")
            for title, count in sorted_titles[:5]:
                print(f"   • {title} ({count}x)")
        
        if os.getenv("DISABLE_GLOSSARY_TRANSLATION", "0") == "1":
            print("📑 Translation disabled - keeping original terms")
            translations = {term: term for term in all_terms}
        else:
            print(f"📑 Translating {len(all_terms)} terms...")
            translations = self._translate_terms_batch(all_terms, language_hint, batch_size)
        
        glossary_entries = {}
        
        for name, _ in sorted_names:
            if name in translations:
                glossary_entries[name] = translations[name]
        
        for title, _ in sorted_titles:
            if title in translations:
                glossary_entries[title] = translations[title]
        
        if existing_glossary and isinstance(existing_glossary, list):
            print("📑 Merging with existing manual glossary...")
            merged_count = 0
            
            for char in existing_glossary:
                original = char.get('original_name', '')
                translated = char.get('name', original)
                
                if original and translated and original not in glossary_entries:
                    glossary_entries[original] = translated
                    merged_count += 1
            
            print(f"📑 Added {merged_count} entries from existing glossary")
        
        glossary_data = {
            "metadata": {
                "language": language_hint,
                "extraction_method": "pattern_based",
                "names_count": len(sorted_names),
                "titles_count": len(sorted_titles),
                "min_frequency": min_frequency,
                "generation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "merged_with_manual": existing_glossary is not None
            },
            "entries": glossary_entries
        }
        
        glossary_path = os.path.join(output_dir, "glossary.json")
        with open(glossary_path, 'w', encoding='utf-8') as f:
            json.dump(glossary_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n📑 ✅ TARGETED GLOSSARY SAVED!")
        print(f"📑 File: {glossary_path}")
        print(f"📑 Names: {len(sorted_names)}, Titles: {len(sorted_titles)}")
        if existing_glossary:
            print(f"📑 Total entries (including manual): {len(glossary_entries)}")
        
        simple_glossary_path = os.path.join(output_dir, "glossary_simple.json")
        with open(simple_glossary_path, 'w', encoding='utf-8') as f:
            json.dump(glossary_entries, f, ensure_ascii=False, indent=2)
        
        return glossary_entries
    
    def _extract_names_for_honorific(self, honorific, all_text, language_hint, 
                                    min_frequency, names_with_honorifics, 
                                    standalone_names, is_valid_name):
        """Extract names for a specific honorific"""
        if language_hint == 'korean' and not honorific.startswith('-'):
            pattern = r'([\uac00-\ud7af]{2,4})(?=' + re.escape(honorific) + r'(?:\s|[,.\!?]|$))'
            
            for match in re.finditer(pattern, all_text):
                potential_name = match.group(1)
                
                if is_valid_name(potential_name, 'korean'):
                    full_form = potential_name + honorific
                    
                    count = len(re.findall(re.escape(full_form) + r'(?=\s|[,.\!?]|$)', all_text))
                    
                    if count >= min_frequency:
                        context_patterns = [
                            full_form + r'[은는이가]',
                            full_form + r'[을를]',
                            full_form + r'[에게한테]',
                            r'["]' + full_form,
                            full_form + r'[,]',
                        ]
                        
                        context_count = 0
                        for ctx_pattern in context_patterns:
                            context_count += len(re.findall(ctx_pattern, all_text))
                        
                        if context_count > 0:
                            names_with_honorifics[full_form] = count
                            standalone_names[potential_name] = count
                            
        elif language_hint == 'japanese' and not honorific.startswith('-'):
            pattern = r'([\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]{2,5})(?=' + re.escape(honorific) + r'(?:\s|[、。！？]|$))'
            
            for match in re.finditer(pattern, all_text):
                potential_name = match.group(1)
                
                if is_valid_name(potential_name, 'japanese'):
                    full_form = potential_name + honorific
                    count = len(re.findall(re.escape(full_form) + r'(?=\s|[、。！？]|$)', all_text))
                    
                    if count >= min_frequency:
                        names_with_honorifics[full_form] = count
                        standalone_names[potential_name] = count
                        
        elif language_hint == 'chinese' and not honorific.startswith('-'):
            pattern = r'([\u4e00-\u9fff]{2,4})(?=' + re.escape(honorific) + r'(?:\s|[，。！？]|$))'
            
            for match in re.finditer(pattern, all_text):
                potential_name = match.group(1)
                
                if is_valid_name(potential_name, 'chinese'):
                    full_form = potential_name + honorific
                    count = len(re.findall(re.escape(full_form) + r'(?=\s|[，。！？]|$)', all_text))
                    
                    if count >= min_frequency:
                        names_with_honorifics[full_form] = count
                        standalone_names[potential_name] = count
                        
        elif honorific.startswith('-') or honorific.startswith(' '):
            is_space_separated = honorific.startswith(' ')
            
            if is_space_separated:
                pattern_english = r'\b([A-Z][a-zA-Z]+)' + re.escape(honorific) + r'(?=\s|[,.\!?]|$)'
            else:
                pattern_english = r'\b([A-Z][a-zA-Z]+)' + re.escape(honorific) + r'\b'
            
            matches = re.finditer(pattern_english, all_text)
            
            for match in matches:
                potential_name = match.group(1)
                
                if is_valid_name(potential_name, 'english'):
                    full_form = potential_name + honorific
                    if is_space_separated:
                        count = len(re.findall(re.escape(full_form) + r'(?=\s|[,.\!?]|$)', all_text))
                    else:
                        count = len(re.findall(re.escape(full_form) + r'\b', all_text))
                    
                    if count >= min_frequency:
                        names_with_honorifics[full_form] = count
                        standalone_names[potential_name] = count
            
            korean_family_names = r'(?:Kim|Lee|Park|Choi|Jung|Jeong|Kang|Cho|Jo|Yoon|Yun|Jang|Chang|Lim|Im|Han|Oh|Seo|Shin|Kwon|Hwang|Ahn|An|Song|Hong|Yoo|Yu|Ko|Go|Moon|Mun|Yang|Bae|Baek|Paek|Nam|Noh|No|Roh|Ha|Heo|Hur|Koo|Ku|Gu|Min|Sim|Shim)'
            
            if is_space_separated:
                pattern_korean = r'\b(' + korean_family_names + r'(?:\s+[A-Z][a-z]+(?:-[A-Z][a-z]+)?)?|[A-Z][a-z]+(?:-[A-Z][a-z]+)+)' + re.escape(honorific) + r'(?=\s|[,.\!?]|$)'
            else:
                pattern_korean = r'\b(' + korean_family_names + r'(?:\s+[A-Z][a-z]+(?:-[A-Z][a-z]+)?)?|[A-Z][a-z]+(?:-[A-Z][a-z]+)+)' + re.escape(honorific) + r'\b'
            
            matches = re.finditer(pattern_korean, all_text, re.IGNORECASE)
            
            for match in matches:
                potential_name = match.group(1)
                potential_name = potential_name.strip()
                
                if 2 <= len(potential_name) <= 30:
                    full_form = potential_name + honorific
                    if is_space_separated:
                        count = len(re.findall(re.escape(full_form) + r'(?=\s|[,.\!?]|$)', all_text, re.IGNORECASE))
                    else:
                        count = len(re.findall(re.escape(full_form) + r'\b', all_text, re.IGNORECASE))
                    
                    if count >= min_frequency:
                        names_with_honorifics[full_form] = count
                        standalone_names[potential_name] = count
    
    def _translate_terms_batch(self, term_list, profile_name, batch_size=50):
        """Use GUI-controlled batch size for translation"""
        if not term_list or os.getenv("DISABLE_GLOSSARY_TRANSLATION", "0") == "1":
            print(f"📑 Glossary translation disabled or no terms to translate")
            return {term: term for term in term_list}
        
        try:
            MODEL = os.getenv("MODEL", "gemini-1.5-flash")
            API_KEY = (os.getenv("API_KEY") or 
                       os.getenv("OPENAI_API_KEY") or 
                       os.getenv("OPENAI_OR_Gemini_API_KEY") or
                       os.getenv("GEMINI_API_KEY"))
            
            if not API_KEY:
                print(f"📑 No API key found, skipping translation")
                return {term: term for term in term_list}
            
            print(f"📑 Translating {len(term_list)} {profile_name} terms to English using batch size {batch_size}...")
            
            from unified_api_client import UnifiedClient
            client = UnifiedClient(model=MODEL, api_key=API_KEY)
            
            all_translations = {}
            
            for i in range(0, len(term_list), batch_size):
                batch = term_list[i:i + batch_size]
                batch_num = (i // batch_size) + 1
                total_batches = (len(term_list) + batch_size - 1) // batch_size
                
                print(f"📑 Processing batch {batch_num}/{total_batches} ({len(batch)} terms)...")
                
                terms_text = ""
                for idx, term in enumerate(batch, 1):
                    terms_text += f"{idx}. {term}\n"
                
                system_prompt = (
                    f"You are translating {profile_name} character names and important terms to English. "
                    "For character names, provide English transliterations or keep as romanized. "
                    "Retain honorifics/suffixes in romaji. "
                    "Keep the same number format in your response."
                )
                
                user_prompt = (
                    f"Translate these {profile_name} character names and terms to English:\n\n"
                    f"{terms_text}\n"
                    "Respond with the same numbered format. For names, provide transliteration or keep romanized."
                )
                
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
                
                try:
                    response, _ = client.send(messages, temperature=0.1, max_tokens=4096)
                    
                    batch_translations = self._parse_translation_response(response, batch)
                    all_translations.update(batch_translations)
                    
                    print(f"📑 Batch {batch_num} completed: {len(batch_translations)} translations")
                    
                    if i + batch_size < len(term_list):
                        time.sleep(0.5)
                        
                except Exception as e:
                    print(f"⚠️ Translation failed for batch {batch_num}: {e}")
                    for term in batch:
                        all_translations[term] = term
            
            for term in term_list:
                if term not in all_translations:
                    all_translations[term] = term
            
            translated_count = sum(1 for term, translation in all_translations.items() 
                                 if translation != term and translation.strip())
            
            print(f"📑 Successfully translated {translated_count}/{len(term_list)} terms")
            return all_translations
            
        except Exception as e:
            print(f"⚠️ Glossary translation failed: {e}")
            return {term: term for term in term_list}
    
    def _parse_translation_response(self, response, original_terms):
        """Parse translation response - handles numbered format"""
        translations = {}
        lines = response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or not line[0].isdigit():
                continue
                
            try:
                number_match = re.match(r'^(\d+)\.?\s*(.+)', line)
                if number_match:
                    num = int(number_match.group(1)) - 1
                    content = number_match.group(2).strip()
                    
                    if 0 <= num < len(original_terms):
                        original_term = original_terms[num]
                        
                        for separator in ['->', '→', ':', '-', '—', '=']:
                            if separator in content:
                                parts = content.split(separator, 1)
                                if len(parts) == 2:
                                    translation = parts[1].strip()
                                    translation = translation.strip('"\'()[]')
                                    if translation and translation != original_term:
                                        translations[original_term] = translation
                                        break
                        else:
                            if content != original_term:
                                translations[original_term] = content
                                
            except (ValueError, IndexError):
                continue
        
        return translations

# =====================================================
# UNIFIED UTILITIES
# =====================================================
def sanitize_resource_filename(filename):
    """Sanitize resource filenames for filesystem compatibility"""
    filename = unicodedata.normalize('NFC', filename)
    
    replacements = {
        '/': '_', '\\': '_', ':': '_', '*': '_',
        '?': '_', '"': '_', '<': '_', '>': '_',
        '|': '_', '\0': '', '\n': '_', '\r': '_'
    }
    
    for old, new in replacements.items():
        filename = filename.replace(old, new)
    
    filename = ''.join(char for char in filename if ord(char) >= 32)
    
    name, ext = os.path.splitext(filename)
    
    if not name:
        name = 'resource'
    
    return name + ext

def make_safe_filename(title, actual_num):
    """Create a safe filename that works across different filesystems"""
    if not title:
        return f"chapter_{actual_num:03d}"
    
    title = unicodedata.normalize('NFC', str(title))
    
    dangerous_chars = {
        '/': '_', '\\': '_', ':': '_', '*': '_', '?': '_',
        '"': '_', '<': '_', '>': '_', '|': '_', '\0': '',
        '\n': ' ', '\r': ' ', '\t': ' '
    }
    
    for old, new in dangerous_chars.items():
        title = title.replace(old, new)
    
    title = ''.join(char for char in title if ord(char) >= 32)
    title = re.sub(r'\s+', '_', title)
    title = title.strip('_.• \t')
    
    if not title or title == '_' * len(title):
        title = f"chapter_{actual_num:03d}"
    
    return title

def get_content_hash(html_content):
    """Create a stable hash of content"""
    return ContentProcessor.get_content_hash(html_content)

def clean_ai_artifacts(text, remove_artifacts=True):
    """Remove AI response artifacts from text"""
    return ContentProcessor.clean_ai_artifacts(text, remove_artifacts)

def clean_memory_artifacts(text):
    """Remove any memory/summary artifacts"""
    return ContentProcessor.clean_memory_artifacts(text)

def emergency_restore_paragraphs(text, original_html=None, verbose=True):
    """Emergency restoration when AI returns wall of text"""
    return ContentProcessor.emergency_restore_paragraphs(text, original_html, verbose)

def is_meaningful_text_content(html_content):
    """Check if chapter has meaningful text beyond just structure"""
    return ContentProcessor.is_meaningful_text_content(html_content)

# =====================================================
# GLOBAL SETTINGS AND FLAGS
# =====================================================
logging.basicConfig(level=logging.DEBUG)

try:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='ignore')
except AttributeError:
    if sys.stdout is None:
        devnull = open(os.devnull, "wb")
        sys.stdout = io.TextIOWrapper(devnull, encoding='utf-8', errors='ignore')
    elif hasattr(sys.stdout, 'buffer'):
        try:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='ignore')
        except:
            pass

_stop_requested = False

def set_stop_flag(value):
    """Set the global stop flag"""
    global _stop_requested
    _stop_requested = value

def is_stop_requested():
    """Check if stop was requested"""
    global _stop_requested
    return _stop_requested

def set_output_redirect(log_callback=None):
    """Redirect print statements to a callback function for GUI integration"""
    if log_callback:
        class CallbackWriter:
            def __init__(self, callback):
                self.callback = callback
                
            def write(self, text):
                if text.strip():
                    self.callback(text.strip())
                    
            def flush(self):
                pass
                
        sys.stdout = CallbackWriter(log_callback)

# =====================================================
# EPUB AND FILE PROCESSING
# =====================================================
def extract_chapter_number_from_filename(filename):
    """Extract chapter number from filename, handling various formats"""
    name_without_ext = os.path.splitext(filename)[0]

    patterns = [
        (r'^(\d+)[_\.]', 'direct_number'),
        (r'^response_(\d+)[_\.]', 'response_prefix'),
        (r'[Cc]hapter[_\s]*(\d+)', 'chapter_word'),
        (r'[Cc]h[_\s]*(\d+)', 'ch_abbreviation'),
        (r'No(\d+)', 'no_prefix'),
        (r'第(\d+)[章话回]', 'chinese_chapter'),
        (r'_(\d+)', 'underscore_suffix'),
        (r'-(\d+)', 'dash_suffix'),
        (r'(\d+)', 'trailing_number'),
    ]

    for pattern, method in patterns:
        match = re.search(pattern, name_without_ext, re.IGNORECASE)
        if match:
            return int(match.group(1)), method

    return None, None

def process_chapter_images(chapter_html: str, actual_num: int, image_translator: ImageTranslator, 
                         check_stop_fn=None) -> Tuple[str, Dict[str, str]]:
    """Process and translate images in a chapter"""
    images = image_translator.extract_images_from_chapter(chapter_html)
    
    if not images:
        return chapter_html, {}
        
    print(f"🖼️ Found {len(images)} images in chapter {actual_num}")
    
    soup = BeautifulSoup(chapter_html, 'html.parser')
    
    image_translations = {}
    translated_count = 0
    
    max_images_per_chapter = int(os.getenv('MAX_IMAGES_PER_CHAPTER', '10'))
    if len(images) > max_images_per_chapter:
        print(f"   ⚠️ Chapter has {len(images)} images - processing first {max_images_per_chapter} only")
        images = images[:max_images_per_chapter]
    
    for idx, img_info in enumerate(images, 1):
        if check_stop_fn and check_stop_fn():
            print("❌ Image translation stopped by user")
            break
            
        img_src = img_info['src']
        
        if img_src.startswith('../'):
            img_path = os.path.join(image_translator.output_dir, img_src[3:])
        elif img_src.startswith('./'):
            img_path = os.path.join(image_translator.output_dir, img_src[2:])
        elif img_src.startswith('/'):
            img_path = os.path.join(image_translator.output_dir, img_src[1:])
        else:
            possible_paths = [
                os.path.join(image_translator.images_dir, os.path.basename(img_src)),
                os.path.join(image_translator.output_dir, img_src),
                os.path.join(image_translator.output_dir, 'images', os.path.basename(img_src)),
                os.path.join(image_translator.output_dir, os.path.basename(img_src)),
                os.path.join(image_translator.output_dir, os.path.dirname(img_src), os.path.basename(img_src))
            ]
            
            img_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    img_path = path
                    print(f"   ✅ Found image at: {path}")
                    break
            
            if not img_path:
                print(f"   ❌ Image not found in any location for: {img_src}")
                print(f"   Tried: {possible_paths}")
                continue
        
        img_path = os.path.normpath(img_path)
        
        if not os.path.exists(img_path):
            print(f"   ⚠️ Image not found: {img_path}")
            print(f"   📁 Images directory: {image_translator.images_dir}")
            print(f"   📁 Output directory: {image_translator.output_dir}")
            print(f"   📁 Working directory: {os.getcwd()}")
            
            if os.path.exists(image_translator.images_dir):
                files = os.listdir(image_translator.images_dir)
                print(f"   📁 Files in images dir: {files[:5]}...")
            continue
        
        print(f"   🔍 Processing image {idx}/{len(images)}: {os.path.basename(img_path)}")
        
        context = ""
        if img_info.get('alt'):
            context += f", Alt text: {img_info['alt']}"
            
        if translated_count > 0:
            delay = float(os.getenv('IMAGE_API_DELAY', '1.0'))
            time.sleep(delay)
            
        translation_result = image_translator.translate_image(img_path, context, check_stop_fn)
        
        if translation_result:
            img_tag = None
            for img in soup.find_all('img'):
                if img.get('src') == img_src:
                    img_tag = img
                    break
            
            if img_tag:
                hide_label = os.getenv("HIDE_IMAGE_TRANSLATION_LABEL", "0") == "1"
                
                container = soup.new_tag('div', **{'class': 'translated-text-only' if hide_label else 'image-with-translation'})
                
                img_tag.replace_with(container)
                
                if not hide_label:
                    container.append(img_tag)
                
                translation_div = soup.new_tag('div', **{'class': 'image-translation'})
                
                if '<div class="image-translation">' in translation_result:
                    trans_soup = BeautifulSoup(translation_result, 'html.parser')
                    trans_content = trans_soup.find('div', class_='image-translation')
                    if trans_content:
                        for element in trans_content.children:
                            if element.name:
                                translation_div.append(element)
                else:
                    trans_p = soup.new_tag('p')
                    trans_p.string = translation_result
                    translation_div.append(trans_p)
                
                container.append(translation_div)
                
                translated_count += 1
                
                trans_filename = f"ch{actual_num:03d}_img{idx:02d}_translation.html"
                trans_filepath = os.path.join(image_translator.translated_images_dir, trans_filename)
                
                with open(trans_filepath, 'w', encoding='utf-8') as f:
                    f.write(f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8"/>
    <title>Chapter {actual_num} - Image {idx} Translation</title>
</head>
<body>
    <h2>Chapter {actual_num} - Image {idx}</h2>
    <p>Original: {os.path.basename(img_path)}</p>
    <hr/>
    {translation_div}
</body>
</html>""")
                
                print(f"   ✅ Saved translation to: {trans_filename}")
            else:
                print(f"   ⚠️ Could not find image tag in HTML for: {img_src}")
    
    if translated_count > 0:
        print(f"   🖼️ Successfully translated {translated_count} images")
        
        prog = image_translator.load_progress()
        if "image_chunks" in prog:
            completed_images = []
            for img_key, img_data in prog["image_chunks"].items():
                if len(img_data["completed"]) == img_data["total"]:
                    completed_images.append(img_key)
            
            for img_key in completed_images:
                del prog["image_chunks"][img_key]
                
            if completed_images:
                image_translator.save_progress(prog)
                print(f"   🧹 Cleaned up progress for {len(completed_images)} completed images")
        
        image_translator.save_translation_log(actual_num, image_translations)
        
        return str(soup), image_translations
    else:
        print(f"   ℹ️ No images were successfully translated")
        
    return chapter_html, {}

def detect_novel_numbering(chapters):
    """Detect if the novel uses 0-based or 1-based chapter numbering"""
    print("[DEBUG] Detecting novel numbering system...")
    
    if not chapters:
        return False
    
    if isinstance(chapters[0], str):
        print("[DEBUG] Text file detected, skipping numbering detection")
        return False
    
    patterns = PatternManager.FILENAME_EXTRACT_PATTERNS
    
    has_chapter_0 = False
    has_chapter_1 = False
    lowest_chapter = float('inf')
    
    for idx, chapter in enumerate(chapters):
        extracted_num = None
        
        if 'original_basename' in chapter:
            basename = chapter['original_basename']
            print(f"[DEBUG] Checking basename: {basename}")
            
            for pattern in patterns:
                match = re.search(pattern, basename)
                if match:
                    extracted_num = int(match.group(1))
                    print(f"[DEBUG] Pattern '{pattern}' matched: {extracted_num}")
                    break
        
        if extracted_num is None and 'filename' in chapter:
            filename = chapter['filename']
            for pattern in patterns:
                match = re.search(pattern, os.path.basename(filename))
                if match:
                    extracted_num = int(match.group(1))
                    print(f"[DEBUG] Filename pattern '{pattern}' matched: {extracted_num}")
                    break
        
        if extracted_num is not None:
            if extracted_num == 0:
                has_chapter_0 = True
                print(f"[DEBUG] ✓ Found chapter 0 at index {idx}")
            elif extracted_num == 1:
                has_chapter_1 = True
            lowest_chapter = min(lowest_chapter, extracted_num)
    
    if not has_chapter_0 and chapters:
        first_chapter = chapters[0]
        if first_chapter.get('num') == 0:
            has_chapter_0 = True
            print("[DEBUG] ✓ First chapter has num=0")
    
    if has_chapter_0:
        print(f"[DEBUG] ✅ 0-based novel detected (found chapter 0)")
        return True
    elif lowest_chapter == 0:
        print(f"[DEBUG] ✅ 0-based novel detected (lowest chapter is 0)")
        return True
    else:
        print(f"[DEBUG] ✅ 1-based novel detected (no chapter 0, lowest chapter: {lowest_chapter})")
        return False

def validate_chapter_continuity(chapters):
    """Validate chapter continuity and warn about issues"""
    if not chapters:
        print("No chapters to translate")
        return
    
    issues = []
    
    chapter_nums = [c['num'] for c in chapters]
    duplicates = [num for num in chapter_nums if chapter_nums.count(num) > 1]
    if duplicates:
        issues.append(f"Duplicate chapter numbers found: {set(duplicates)}")
    
    min_num = min(chapter_nums)
    max_num = max(chapter_nums)
    expected = set(range(min_num, max_num + 1))
    actual = set(chapter_nums)
    missing = expected - actual
    if missing:
        issues.append(f"Missing chapter numbers: {sorted(missing)}")
    
    for i in range(len(chapters) - 1):
        for j in range(i + 1, len(chapters)):
            title1 = chapters[i]['title'].lower()
            title2 = chapters[j]['title'].lower()
            if title1 == title2 and chapters[i]['num'] != chapters[j]['num']:
                issues.append(f"Chapters {chapters[i]['num']} and {chapters[j]['num']} have identical titles")
    
    if issues:
        print("\n⚠️  Chapter Validation Issues:")
        for issue in issues:
            print(f"  - {issue}")
        print()

def validate_epub_structure(output_dir):
    """Validate that all necessary EPUB structure files are present"""
    print("🔍 Validating EPUB structure...")
    
    required_files = {
        'container.xml': 'META-INF container file (critical)',
        '*.opf': 'OPF package file (critical)',
        '*.ncx': 'Navigation file (recommended)'
    }
    
    found_files = {}
    missing_files = []
    
    container_path = os.path.join(output_dir, 'container.xml')
    if os.path.exists(container_path):
        found_files['container.xml'] = 'Found'
        print("   ✅ container.xml - Found")
    else:
        missing_files.append('container.xml')
        print("   ❌ container.xml - Missing (CRITICAL)")
    
    opf_files = []
    ncx_files = []
    
    for file in os.listdir(output_dir):
        if file.lower().endswith('.opf'):
            opf_files.append(file)
        elif file.lower().endswith('.ncx'):
            ncx_files.append(file)
    
    if opf_files:
        found_files['opf'] = opf_files
        print(f"   ✅ OPF file(s) - Found: {', '.join(opf_files)}")
    else:
        missing_files.append('*.opf')
        print("   ❌ OPF file - Missing (CRITICAL)")
    
    if ncx_files:
        found_files['ncx'] = ncx_files
        print(f"   ✅ NCX file(s) - Found: {', '.join(ncx_files)}")
    else:
        missing_files.append('*.ncx')
        print("   ⚠️ NCX file - Missing (navigation may not work)")
    
    html_files = [f for f in os.listdir(output_dir) if f.lower().endswith('.html') and f.startswith('response_')]
    if html_files:
        print(f"   ✅ Translated chapters - Found: {len(html_files)} files")
    else:
        print("   ⚠️ No translated chapter files found")
    
    critical_missing = [f for f in missing_files if f in ['container.xml', '*.opf']]
    
    if not critical_missing:
        print("✅ EPUB structure validation PASSED")
        print("   All critical files present for EPUB reconstruction")
        return True
    else:
        print("❌ EPUB structure validation FAILED")
        print(f"   Missing critical files: {', '.join(critical_missing)}")
        print("   EPUB reconstruction may fail without these files")
        return False

def check_epub_readiness(output_dir):
    """Check if the output directory is ready for EPUB compilation"""
    print("📋 Checking EPUB compilation readiness...")
    
    issues = []
    
    if not validate_epub_structure(output_dir):
        issues.append("Missing critical EPUB structure files")
    
    html_files = [f for f in os.listdir(output_dir) if f.lower().endswith('.html') and f.startswith('response_')]
    if not html_files:
        issues.append("No translated chapter files found")
    else:
        print(f"   ✅ Found {len(html_files)} translated chapters")
    
    metadata_path = os.path.join(output_dir, 'metadata.json')
    if os.path.exists(metadata_path):
        print("   ✅ Metadata file present")
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            if 'title' not in metadata:
                issues.append("Metadata missing title")
        except Exception as e:
            issues.append(f"Metadata file corrupted: {e}")
    else:
        issues.append("Missing metadata.json file")
    
    resource_dirs = ['css', 'fonts', 'images']
    found_resources = 0
    for res_dir in resource_dirs:
        res_path = os.path.join(output_dir, res_dir)
        if os.path.exists(res_path):
            files = [f for f in os.listdir(res_path) if os.path.isfile(os.path.join(res_path, f))]
            if files:
                found_resources += len(files)
                print(f"   ✅ Found {len(files)} {res_dir} files")
    
    if found_resources > 0:
        print(f"   ✅ Total resources: {found_resources} files")
    else:
        print("   ⚠️ No resource files found (this may be normal)")
    
    if not issues:
        print("🎉 EPUB compilation readiness: READY")
        print("   All necessary files present for EPUB creation")
        return True
    else:
        print("⚠️ EPUB compilation readiness: ISSUES FOUND")
        for issue in issues:
            print(f"   • {issue}")
        return False

def cleanup_previous_extraction(output_dir):
    """Clean up any files from previous extraction runs"""
    cleanup_items = [
        'css', 'fonts', 'images',
        '.resources_extracted'
    ]
    
    epub_structure_files = [
        'container.xml', 'content.opf', 'toc.ncx'
    ]
    
    cleaned_count = 0
    
    for item in cleanup_items:
        if item.startswith('.'):
            continue
        item_path = os.path.join(output_dir, item)
        try:
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
                print(f"🧹 Removed directory: {item}")
                cleaned_count += 1
        except Exception as e:
            print(f"⚠️ Could not remove directory {item}: {e}")
    
    for epub_file in epub_structure_files:
        file_path = os.path.join(output_dir, epub_file)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"🧹 Removed EPUB file: {epub_file}")
                cleaned_count += 1
        except Exception as e:
            print(f"⚠️ Could not remove {epub_file}: {e}")
    
    try:
        for file in os.listdir(output_dir):
            if file.lower().endswith(('.opf', '.ncx')):
                file_path = os.path.join(output_dir, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"🧹 Removed EPUB file: {file}")
                    cleaned_count += 1
    except Exception as e:
        print(f"⚠️ Error scanning for EPUB files: {e}")
    
    marker_path = os.path.join(output_dir, '.resources_extracted')
    try:
        if os.path.isfile(marker_path):
            os.remove(marker_path)
            print(f"🧹 Removed extraction marker")
            cleaned_count += 1
    except Exception as e:
        print(f"⚠️ Could not remove extraction marker: {e}")
    
    if cleaned_count > 0:
        print(f"🧹 Cleaned up {cleaned_count} items from previous runs")
    
    return cleaned_count

# =====================================================
# API AND TRANSLATION UTILITIES
# =====================================================
def send_with_interrupt(messages, client, temperature, max_tokens, stop_check_fn, chunk_timeout=None):
    """Send API request with interrupt capability and optional timeout retry"""
    result_queue = queue.Queue()
    
    def api_call():
        try:
            start_time = time.time()
            result = client.send(messages, temperature=temperature, max_tokens=max_tokens)
            elapsed = time.time() - start_time
            result_queue.put((result, elapsed))
        except Exception as e:
            result_queue.put(e)
    
    api_thread = threading.Thread(target=api_call)
    api_thread.daemon = True
    api_thread.start()
    
    timeout = chunk_timeout if chunk_timeout is not None else 86400
    check_interval = 0.5
    elapsed = 0
    
    while elapsed < timeout:
        try:
            result = result_queue.get(timeout=check_interval)
            if isinstance(result, Exception):
                raise result
            if isinstance(result, tuple):
                api_result, api_time = result
                if chunk_timeout and api_time > chunk_timeout:
                    raise UnifiedClientError(f"API call took {api_time:.1f}s (timeout: {chunk_timeout}s)")
                return api_result
            return result
        except queue.Empty:
            if stop_check_fn():
                raise UnifiedClientError("Translation stopped by user")
            elapsed += check_interval
    
    raise UnifiedClientError(f"API call timed out after {timeout} seconds")

def parse_token_limit(env_value):
    """Parse token limit from environment variable"""
    if not env_value or env_value.strip() == "":
        return None, "unlimited"
    
    env_value = env_value.strip()
    if env_value.lower() == "unlimited":
        return None, "unlimited"
    
    if env_value.isdigit() and int(env_value) > 0:
        limit = int(env_value)
        return limit, str(limit)
    
    return 1000000, "1000000 (default)"

def build_system_prompt(user_prompt, glossary_path):
    """Build the system prompt with glossary - NO FALLBACK"""
    append_glossary = os.getenv("APPEND_GLOSSARY", "1") == "1"
    
    system = user_prompt if user_prompt else ""
    
    def format_glossary_for_prompt(glossary_data):
        """Convert various glossary formats into a unified prompt format"""
        formatted_entries = {}
        
        try:
            if isinstance(glossary_data, list):
                for char in glossary_data:
                    if not isinstance(char, dict):
                        continue
                        
                    original = char.get('original_name', '')
                    translated = char.get('name', original)
                    if original and translated:
                        formatted_entries[original] = translated
                    
                    title = char.get('title')
                    if title and original:
                        formatted_entries[f"{original} ({title})"] = f"{translated} ({title})"
                    
                    refer_map = char.get('how_they_refer_to_others', {})
                    if isinstance(refer_map, dict):
                        for other_name, reference in refer_map.items():
                            if other_name and reference:
                                formatted_entries[f"{original} → {other_name}"] = f"{translated} → {reference}"
            
            elif isinstance(glossary_data, dict):
                if "entries" in glossary_data and isinstance(glossary_data["entries"], dict):
                    formatted_entries = glossary_data["entries"]
                elif all(isinstance(k, str) and isinstance(v, str) for k, v in glossary_data.items() if k != "metadata"):
                    formatted_entries = {k: v for k, v in glossary_data.items() if k != "metadata"}
            
            return formatted_entries
            
        except Exception as e:
            print(f"Warning: Error formatting glossary: {e}")
            return {}
    
    if append_glossary and glossary_path and os.path.exists(glossary_path):
        try:
            with open(glossary_path, "r", encoding="utf-8") as gf:
                glossary_data = json.load(gf)
            
            formatted_entries = format_glossary_for_prompt(glossary_data)
            
            if formatted_entries:
                if isinstance(formatted_entries, dict):
                    glossary_block = json.dumps(formatted_entries, ensure_ascii=False, indent=2)
                    if system:
                        system += "\n\n"
                    system += (
                        "Character/Term Glossary (use these translations consistently):\n"
                        f"{glossary_block}"
                    )
        except Exception as e:
            print(f"Warning: Could not load glossary: {e}")
    
    return system

def translate_title(title, client, system_prompt, user_prompt, temperature=0.3):
    """Translate the book title using the configured settings"""
    if not title or not title.strip():
        return title
        
    print(f"📚 Processing book title: {title}")
    
    try:
        if os.getenv("TRANSLATE_BOOK_TITLE", "1") == "0":
            print(f"📚 Book title translation disabled - keeping original")
            return title
        
        book_title_prompt = os.getenv("BOOK_TITLE_PROMPT", 
            "Translate this book title to English while retaining any acronyms:")
        
        messages = [
            {"role": "system", "content": "You are a translator. Respond with only the translated text, nothing else. Do not add any explanation or additional content."},
            {"role": "user", "content": f"{book_title_prompt}\n\n{title}"}
        ]
        
        translated_title, _ = client.send(messages, temperature=temperature, max_tokens=512)
        
        print(f"[DEBUG] Raw API response: '{translated_title}'")
        print(f"[DEBUG] Response length: {len(translated_title)} (original: {len(title)})")
        newline = '\n'
        print(f"[DEBUG] Has newlines: {repr(translated_title) if newline in translated_title else 'No'}")
        
        translated_title = translated_title.strip()
        
        if ((translated_title.startswith('"') and translated_title.endswith('"')) or 
            (translated_title.startswith("'") and translated_title.endswith("'"))):
            translated_title = translated_title[1:-1].strip()
        
        if '\n' in translated_title:
            print(f"⚠️ API returned multi-line content, keeping original title")
            return title           
            
        if any(char in translated_title for char in ['{', '}', '[', ']', '"role":', '"content":']):
            print(f"⚠️ API returned structured content, keeping original title")
            return title
            
        if any(tag in translated_title.lower() for tag in ['<p>', '</p>', '<h1>', '</h1>', '<html']):
            print(f"⚠️ API returned HTML content, keeping original title")
            return title
        
        print(f"✅ Processed title: {translated_title}")
        return translated_title
        
    except Exception as e:
        print(f"⚠️ Failed to process title: {e}")
        return title

# =====================================================
# MAIN TRANSLATION FUNCTION
# =====================================================
def main(log_callback=None, stop_callback=None):
    """Main translation function with enhanced duplicate detection and progress tracking"""
    
    config = TranslationConfig()
    
    args = None
    chapters_completed = 0
    chunks_completed = 0
    
    input_path = config.input_path
    if not input_path and len(sys.argv) > 1:
        input_path = sys.argv[1]
    
    is_text_file = input_path.lower().endswith('.txt')
    
    import json as _json
    _original_load = _json.load
      
    def debug_json_load(fp, *args, **kwargs):
        result = _original_load(fp, *args, **kwargs)
        if isinstance(result, list) and len(result) > 0:
            if isinstance(result[0], dict) and 'original_name' in result[0]:
                print(f"[DEBUG] Loaded glossary list with {len(result)} items from {fp.name if hasattr(fp, 'name') else 'unknown'}")
        return result
    
    _json.load = debug_json_load
    
    if log_callback:
        set_output_redirect(log_callback)
    
    def check_stop():
        if stop_callback and stop_callback():
            print("❌ Translation stopped by user request.")
            return True
        return is_stop_requested()
    
    if config.EMERGENCY_RESTORE:
        print("✅ Emergency paragraph restoration is ENABLED")
    else:
        print("⚠️ Emergency paragraph restoration is DISABLED")
    
    print(f"[DEBUG] REMOVE_AI_ARTIFACTS environment variable: {os.getenv('REMOVE_AI_ARTIFACTS', 'NOT SET')}")
    print(f"[DEBUG] REMOVE_AI_ARTIFACTS parsed value: {config.REMOVE_AI_ARTIFACTS}")
    if config.REMOVE_AI_ARTIFACTS:
        print("⚠️ AI artifact removal is ENABLED - will clean AI response artifacts")
    else:
        print("✅ AI artifact removal is DISABLED - preserving all content as-is")
       
    if '--epub' in sys.argv or (len(sys.argv) > 1 and sys.argv[1].endswith(('.epub', '.txt'))):
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('epub', help='Input EPUB or text file')
        args = parser.parse_args()
        input_path = args.epub
    
    is_text_file = input_path.lower().endswith('.txt')
    
    if is_text_file:
        file_base = os.path.splitext(os.path.basename(input_path))[0]
    else:
        epub_base = os.path.splitext(os.path.basename(input_path))[0]
        file_base = epub_base
        
    out = file_base
    os.makedirs(out, exist_ok=True)
    print(f"[DEBUG] Created output folder → {out}")
    
    cleanup_previous_extraction(out)

    os.environ["EPUB_OUTPUT_DIR"] = out
    payloads_dir = out
    
    history_manager = HistoryManager(payloads_dir)
    chapter_splitter = ChapterSplitter(model_name=config.MODEL)
    chunk_context_manager = ChunkContextManager()
    progress_manager = ProgressManager(payloads_dir)
    chapter_extractor = ChapterExtractor()
    glossary_manager = GlossaryManager()

    history_file = os.path.join(payloads_dir, "translation_history.json")
    if os.path.exists(history_file):
        os.remove(history_file)
        print(f"[DEBUG] Purged translation history → {history_file}")

    print("🔍 Checking for deleted output files...")
    progress_manager.cleanup_missing_files(out)
    progress_manager.save()

    if os.getenv("RESET_FAILED_CHAPTERS", "1") == "1":
        reset_count = 0
        for chapter_key, chapter_info in list(progress_manager.prog["chapters"].items()):
            status = chapter_info.get("status")
            
            if status in ["failed", "qa_failed", "file_missing", "error", "file_deleted"]:
                del progress_manager.prog["chapters"][chapter_key]
                
                content_hash = chapter_info.get("content_hash")
                if content_hash and content_hash in progress_manager.prog["content_hashes"]:
                    del progress_manager.prog["content_hashes"][content_hash]
                
                if chapter_key in progress_manager.prog.get("chapter_chunks", {}):
                    del progress_manager.prog["chapter_chunks"][chapter_key]
                    
                reset_count += 1
        
        if reset_count > 0:
            print(f"🔄 Reset {reset_count} failed/deleted chapters for re-translation")
            progress_manager.save()

    if check_stop():
        return

    if not config.API_KEY:
        print("❌ Error: Set API_KEY, OPENAI_API_KEY, or OPENAI_OR_Gemini_API_KEY in your environment.")
        return

    print(f"[DEBUG] Found API key: {config.API_KEY[:10]}...")
    print(f"[DEBUG] Using model = {config.MODEL}")
    print(f"[DEBUG] Max output tokens = {config.MAX_OUTPUT_TOKENS}")

    client = UnifiedClient(model=config.MODEL, api_key=config.API_KEY)
    
    if is_text_file:
        print("📄 Processing text file...")
        try:
            txt_processor = TextFileProcessor(input_path, out)
            chapters = txt_processor.extract_chapters()
            txt_processor.save_original_structure()
            
            metadata = {
                "title": os.path.splitext(os.path.basename(input_path))[0],
                "type": "text",
                "chapter_count": len(chapters)
            }
        except ImportError as e:
            print(f"❌ Error: Text file processor not available: {e}")
            if log_callback:
                log_callback(f"❌ Error: Text file processor not available: {e}")
            return
        except Exception as e:
            print(f"❌ Error processing text file: {e}")
            if log_callback:
                log_callback(f"❌ Error processing text file: {e}")
            return
    else:
        print("🚀 Using comprehensive chapter extraction with resource handling...")
        with zipfile.ZipFile(input_path, 'r') as zf:
            metadata = chapter_extractor._extract_epub_metadata(zf)
            chapters = chapter_extractor.extract_chapters(zf, out)
            
            validate_chapter_continuity(chapters)
        
        print("\n" + "="*50)
        validate_epub_structure(out)
        print("="*50 + "\n")
    
    progress_manager.migrate_to_content_hash(chapters)
    progress_manager.save()

    if check_stop():
        return

    metadata_path = os.path.join(out, "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r', encoding='utf-8') as mf:
            metadata = json.load(mf)

    metadata["chapter_count"] = len(chapters)
    metadata["chapter_titles"] = {str(c["num"]): c["title"] for c in chapters}

    print(f"[DEBUG] Initializing client with model = {config.MODEL}")
    client = UnifiedClient(model=config.MODEL, api_key=config.API_KEY)
    
    if "title" in metadata and config.TRANSLATE_BOOK_TITLE and not metadata.get("title_translated", False):
        original_title = metadata["title"]
        print(f"📚 Original title: {original_title}")
        
        if not check_stop():
            translated_title = translate_title(
                original_title, 
                client, 
                None,
                None,
                config.TEMP
            )
            
            metadata["original_title"] = original_title
            metadata["title"] = translated_title
            metadata["title_translated"] = True
            
            print(f"📚 Translated title: {translated_title}")
        else:
            print("❌ Title translation skipped due to stop request")
    
    with open(metadata_path, 'w', encoding='utf-8') as mf:
        json.dump(metadata, mf, ensure_ascii=False, indent=2)
    print(f"💾 Saved metadata with {'translated' if metadata.get('title_translated', False) else 'original'} title")
        
    print("\n" + "="*50)
    print("📑 GLOSSARY GENERATION PHASE")
    print("="*50)

    if config.MANUAL_GLOSSARY and os.path.isfile(config.MANUAL_GLOSSARY):
        target_path = os.path.join(out, "glossary.json")
        if os.path.abspath(config.MANUAL_GLOSSARY) != os.path.abspath(target_path):
            shutil.copy(config.MANUAL_GLOSSARY, target_path)
            print("📑 Using manual glossary from:", config.MANUAL_GLOSSARY)
        else:
            print("📑 Using existing glossary:", config.MANUAL_GLOSSARY)
    elif not config.DISABLE_AUTO_GLOSSARY:
        print("📑 Starting automatic glossary generation...")
        
        try:
            instructions = ""
            glossary_manager.save_glossary(out, chapters, instructions)
            print("✅ Automatic glossary generation COMPLETED")
        except Exception as e:
            print(f"❌ Glossary generation failed: {e}")
    else:
        print("📑 Automatic glossary disabled - no glossary will be used")
        with open(os.path.join(out, "glossary.json"), 'w', encoding='utf-8') as f:
            json.dump({}, f, ensure_ascii=False, indent=2)

    glossary_path = os.path.join(out, "glossary.json")
    if os.path.exists(glossary_path):
        try:
            with open(glossary_path, 'r', encoding='utf-8') as f:
                glossary_data = json.load(f)
            
            if isinstance(glossary_data, dict):
                if 'entries' in glossary_data and isinstance(glossary_data['entries'], dict):
                    entry_count = len(glossary_data['entries'])
                    sample_items = list(glossary_data['entries'].items())[:3]
                else:
                    entry_count = len(glossary_data)
                    sample_items = list(glossary_data.items())[:3]
                
                print(f"📑 Glossary ready with {entry_count} entries")
                print("📑 Sample glossary entries:")
                for key, value in sample_items:
                    print(f"   • {key} → {value}")
                    
            elif isinstance(glossary_data, list):
                print(f"📑 Glossary ready with {len(glossary_data)} entries")
                print("📑 Sample glossary entries:")
                for i, entry in enumerate(glossary_data[:3]):
                    if isinstance(entry, dict):
                        original = entry.get('original_name', '?')
                        translated = entry.get('name', original)
                        print(f"   • {original} → {translated}")
            else:
                print(f"⚠️ Unexpected glossary format: {type(glossary_data)}")
                
        except Exception as e:
            print(f"⚠️ Glossary file exists but is corrupted: {e}")
            with open(glossary_path, 'w', encoding='utf-8') as f:
                json.dump({}, f, ensure_ascii=False, indent=2)
    else:
        print("⚠️ No glossary file found, creating empty one")
        with open(glossary_path, 'w', encoding='utf-8') as f:
            json.dump({}, f, ensure_ascii=False, indent=2)

    print("="*50)
    print("🚀 STARTING MAIN TRANSLATION PHASE")
    print("="*50 + "\n")

    glossary_path = os.path.join(out, "glossary.json")
    if os.path.exists(glossary_path):
        try:
            with open(glossary_path, 'r', encoding='utf-8') as f:
                g_data = json.load(f)
            print(f"[DEBUG] Glossary type before translation: {type(g_data)}")
            if isinstance(g_data, list):
                print(f"[DEBUG] Glossary is a list")
        except Exception as e:
            print(f"[DEBUG] Error checking glossary: {e}")
            
    system = build_system_prompt(config.SYSTEM_PROMPT, glossary_path)
    base_msg = [{"role": "system", "content": system}]
    
    image_translator = None

    if config.ENABLE_IMAGE_TRANSLATION:
        print(f"🖼️ Image translation enabled for model: {config.MODEL}")
        print("🖼️ Image translation will use your custom system prompt and glossary")
        image_translator = ImageTranslator(
            client, 
            out, 
            config.PROFILE_NAME, 
            system, 
            config.TEMP,
            log_callback 
        )
        
        known_vision_models = [
            'gemini-1.5-pro', 'gemini-1.5-flash', 'gemini-2.0-flash', 'gemini-2.0-flash-exp', 
            'gpt-4-turbo', 'gpt-4o', 'gpt-4.1-mini', 'gpt-4.1-nano', 'o4-mini'
        ]
        
        if config.MODEL.lower() not in known_vision_models:
            print(f"⚠️ Note: {config.MODEL} may not have vision capabilities. Image translation will be attempted anyway.")
    else:
        print("ℹ️ Image translation disabled by user")
    
    total_chapters = len(chapters)

    if chapters:
        uses_zero_based = detect_novel_numbering(chapters)
    else:
        uses_zero_based = False

    rng = os.getenv("CHAPTER_RANGE", "")
    start = None
    end = None
    if rng and re.match(r"^\d+\s*-\s*\d+$", rng):
        start, end = map(int, rng.split("-", 1))
        
        if config.DISABLE_ZERO_DETECTION:
            print(f"📊 0-based detection disabled - using range as specified: {start}-{end}")
            uses_zero_based = False
        else:
            if chapters:
                uses_zero_based = detect_novel_numbering(chapters)
            else:
                uses_zero_based = False
                
            if uses_zero_based:
                print(f"📊 0-based novel detected")
                print(f"📊 User range {start}-{end} will map to files {start-1}-{end-1}")
            else:
                print(f"📊 1-based novel detected")
                print(f"📊 Using range as specified: {start}-{end}")
    
    print("📊 Calculating total chunks needed...")
    total_chunks_needed = 0
    chunks_per_chapter = {}
    chapters_to_process = 0

    for idx, c in enumerate(chapters):
        chap_num = c["num"]
        content_hash = c.get("content_hash") or ContentProcessor.get_content_hash(c["body"])
        
        actual_num = FileUtilities.extract_actual_chapter_number(c)
        
        c['actual_chapter_num'] = actual_num

        if start is not None:
            if not (start <= actual_num <= end):
                print(f"[SKIP] Chapter {actual_num} outside range {start}-{end}")
                continue
                
        needs_translation, skip_reason, _ = progress_manager.check_chapter_status(
            idx, actual_num, content_hash, out
        )
        
        if not needs_translation:
            chunks_per_chapter[idx] = 0
            continue
        
        chapters_to_process += 1
        
        chapter_key = str(idx)
        if chapter_key in progress_manager.prog["chapters"] and progress_manager.prog["chapters"][chapter_key].get("status") == "in_progress":
            pass
        
        _tok_env = os.getenv("MAX_INPUT_TOKENS", "1000000").strip()
        max_tokens_limit, _ = parse_token_limit(_tok_env)
        
        system_tokens = chapter_splitter.count_tokens(system)
        history_tokens = config.HIST_LIMIT * 2 * 1000
        safety_margin = 1000
        
        if max_tokens_limit is not None:
            available_tokens = max_tokens_limit - system_tokens - history_tokens - safety_margin
            chunks = chapter_splitter.split_chapter(c["body"], available_tokens)
        else:
            chunks = [(c["body"], 1, 1)]
        
        chapter_key_str = content_hash
        old_key_str = str(idx)

        if chapter_key_str not in progress_manager.prog.get("chapter_chunks", {}) and old_key_str in progress_manager.prog.get("chapter_chunks", {}):
            progress_manager.prog["chapter_chunks"][chapter_key_str] = progress_manager.prog["chapter_chunks"][old_key_str]
            del progress_manager.prog["chapter_chunks"][old_key_str]
            print(f"[PROGRESS] Migrated chunks for chapter {actual_num} to new tracking system")

        if chapter_key_str in progress_manager.prog.get("chapter_chunks", {}):
            completed_chunks = len(progress_manager.prog["chapter_chunks"][chapter_key_str].get("completed", []))
            chunks_needed = len(chunks) - completed_chunks
            chunks_per_chapter[idx] = max(0, chunks_needed)
        else:
            chunks_per_chapter[idx] = len(chunks)
        
        total_chunks_needed += chunks_per_chapter[idx]
    
    print(f"📊 Total chunks to translate: {total_chunks_needed}")
    print(f"📚 Chapters to process: {chapters_to_process}")
    
    multi_chunk_chapters = [(idx, count) for idx, count in chunks_per_chapter.items() if count > 1]
    if multi_chunk_chapters:
        print("📄 Chapters requiring multiple chunks:")
        for idx, chunk_count in multi_chunk_chapters:
            chap = chapters[idx]
            print(f"   • Chapter {idx+1} ({chap['title'][:30]}...): {chunk_count} chunks")
    
    translation_start_time = time.time()
    chunks_completed = 0
    chapters_completed = 0
    
    current_chunk_number = 0

    if config.BATCH_TRANSLATION:
        print(f"\n📦 PARALLEL TRANSLATION MODE ENABLED")
        print(f"📦 Processing chapters with up to {config.BATCH_SIZE} concurrent API calls")
        
        import concurrent.futures
        from threading import Lock
        
        progress_lock = Lock()
        
        chapters_to_translate = []
        
        for idx, c in enumerate(chapters):
            chap_num = c["num"]
            content_hash = c.get("content_hash") or ContentProcessor.get_content_hash(c["body"])
            
            if start is not None:
                actual_num = FileUtilities.extract_actual_chapter_number(c)
                
                if not (start <= actual_num <= end):
                    continue
            
            needs_translation, skip_reason, existing_file = progress_manager.check_chapter_status(
                idx, chap_num, content_hash, out
            )
            
            if not needs_translation:
                print(f"[SKIP] {skip_reason}")
                continue
            
            has_images = c.get('has_images', False)
            has_meaningful_text = ContentProcessor.is_meaningful_text_content(c["body"])
            text_size = c.get('file_size', 0)
            
            is_empty_chapter = (not has_images and text_size < 50)
            is_image_only_chapter = (has_images and not has_meaningful_text)
            
            if is_empty_chapter:
                print(f"📄 Empty chapter {chap_num} - will process individually")
                
                safe_title = make_safe_filename(c['title'], c['num'])
                
                if isinstance(c['num'], float):
                    fname = f"response_{c['num']:06.1f}_{safe_title}.html"
                else:
                    fname = f"response_{c['num']:03d}_{safe_title}.html"
                with open(os.path.join(out, fname), 'w', encoding='utf-8') as f:
                    f.write(c["body"])
                progress_manager.update(idx, chap_num, content_hash, fname, status="completed_empty")
                progress_manager.save()
                chapters_completed += 1
                continue
            
            chapters_to_translate.append((idx, c))
        
        print(f"📊 Found {len(chapters_to_translate)} chapters to translate in parallel")
        
        batch_processor = BatchTranslationProcessor(
            config, client, base_msg, out, progress_lock,
            progress_manager.save, 
            lambda idx, actual_num, content_hash, output_filename=None, status="completed": progress_manager.update(idx, actual_num, content_hash, output_filename, status),
            check_stop,
            image_translator
        )
        
        total_to_process = len(chapters_to_translate)
        processed = 0
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=config.BATCH_SIZE) as executor:
            for batch_start in range(0, total_to_process, config.BATCH_SIZE * 3):
                if check_stop():
                    print("❌ Translation stopped during parallel processing")
                    executor.shutdown(wait=False)
                    return
                
                batch_end = min(batch_start + config.BATCH_SIZE * 3, total_to_process)
                current_batch = chapters_to_translate[batch_start:batch_end]
                
                print(f"\n📦 Submitting batch {batch_start//config.BATCH_SIZE + 1}: {len(current_batch)} chapters")
                
                future_to_chapter = {
                    executor.submit(batch_processor.process_single_chapter, chapter_data): chapter_data
                    for chapter_data in current_batch
                }
                
                active_count = 0
                completed_in_batch = 0
                failed_in_batch = 0
                
                for future in concurrent.futures.as_completed(future_to_chapter):
                    if check_stop():
                        print("❌ Translation stopped")
                        executor.shutdown(wait=False)
                        return
                    
                    chapter_data = future_to_chapter[future]
                    idx, chapter = chapter_data
                    
                    try:
                        success, chap_num = future.result()
                        if success:
                            completed_in_batch += 1
                            print(f"✅ Chapter {chap_num} done ({completed_in_batch + failed_in_batch}/{len(current_batch)} in batch)")
                        else:
                            failed_in_batch += 1
                            print(f"❌ Chapter {chap_num} failed ({completed_in_batch + failed_in_batch}/{len(current_batch)} in batch)")
                    except Exception as e:
                        failed_in_batch += 1
                        print(f"❌ Chapter thread error: {e}")
                    
                    processed += 1
                    
                    progress_percent = (processed / total_to_process) * 100
                    print(f"📊 Overall Progress: {processed}/{total_to_process} ({progress_percent:.1f}%)")
                
                print(f"\n📦 Batch Summary:")
                print(f"   ✅ Successful: {completed_in_batch}")
                print(f"   ❌ Failed: {failed_in_batch}")
                
                if batch_end < total_to_process:
                    print(f"⏳ Waiting {config.DELAY}s before next batch...")
                    time.sleep(config.DELAY)
        
        chapters_completed = batch_processor.chapters_completed
        chunks_completed = batch_processor.chunks_completed
        
        print(f"\n🎉 Parallel translation complete!")
        print(f"   Total chapters processed: {processed}")
        print(f"   Successful: {chapters_completed}")
        
        config.BATCH_TRANSLATION = False

    if not config.BATCH_TRANSLATION:
        translation_processor = TranslationProcessor(config, client, out, log_callback, check_stop)
        
        for idx, c in enumerate(chapters):
            actual_num = FileUtilities.extract_actual_chapter_number(c)
            c['actual_chapter_num'] = actual_num

        for idx, c in enumerate(chapters):
            chap_num = c["num"]
            actual_num = c['actual_chapter_num']
            content_hash = c.get("content_hash") or ContentProcessor.get_content_hash(c["body"])
            
            if start is not None and not (start <= actual_num <= end):
                print(f"[SKIP] Chapter {actual_num} (file: {c.get('original_basename', 'unknown')}) outside range {start}-{end}")
                continue
            
            needs_translation, skip_reason, existing_file = progress_manager.check_chapter_status(
                idx, actual_num, content_hash, out
            )
            
            if not needs_translation:
                print(f"[SKIP] {skip_reason}")
                continue

            chapter_position = f"{chapters_completed + 1}/{chapters_to_process}"
          
            print(f"\n🔄 Processing #{idx+1}/{total_chapters} (Actual: Chapter {actual_num}) ({chapter_position} to translate): {c['title']} [File: {c.get('original_basename', f'Chapter_{chap_num}')}]")

            chunk_context_manager.start_chapter(chap_num, c['title'])
            
            has_images = c.get('has_images', False)
            has_meaningful_text = ContentProcessor.is_meaningful_text_content(c["body"])
            text_size = c.get('file_size', 0)
            
            is_empty_chapter = (not has_images and text_size < 50)
            is_image_only_chapter = (has_images and not has_meaningful_text)
            is_mixed_content = (has_images and has_meaningful_text)
            is_text_only = (not has_images and has_meaningful_text)
            
            if is_empty_chapter:
                print(f"📄 Empty chapter detected")
            elif is_image_only_chapter:
                if image_translator and config.ENABLE_IMAGE_TRANSLATION:
                    print(f"🖼️ Translating {c.get('image_count', 0)} images...")
                    image_translator.set_current_chapter(chap_num)
                    translated_html, image_translations = process_chapter_images(
                        c["body"], 
                        actual_num, 
                        image_translator,
                        check_stop
                    )
                print(f"📸 Image-only chapter: {c.get('image_count', 0)} images, no meaningful text")
            elif is_mixed_content:
                print(f"📖📸 Mixed content: {text_size} chars + {c.get('image_count', 0)} images")
            else:
                print(f"📖 Text-only chapter: {text_size} characters")

            if is_empty_chapter:
                print(f"📄 Copying empty chapter as-is")
                safe_title = make_safe_filename(c['title'], c['num'])
                if isinstance(c['num'], float):
                    fname = f"response_{c['num']:06.1f}_{safe_title}.html"
                else:
                    fname = f"response_{c['num']:03d}_{safe_title}.html"
                
                with open(os.path.join(out, fname), 'w', encoding='utf-8') as f:
                    f.write(c["body"])
                
                print(f"[Chapter {idx+1}/{total_chapters}] ✅ Saved empty chapter")
                progress_manager.update(idx, chap_num, content_hash, fname, status="completed_empty")
                progress_manager.save()
                chapters_completed += 1
                continue

            elif is_image_only_chapter:
                if image_translator and config.ENABLE_IMAGE_TRANSLATION:
                    print(f"🖼️ Translating {c.get('image_count', 0)} images...")
                    translated_html, image_translations = process_chapter_images(
                        c["body"], 
                        chap_num, 
                        image_translator,
                        check_stop
                    )
                    
                    if '<div class="image-translation">' in translated_html:
                        print(f"✅ Successfully translated images")
                        status = "completed"
                    else:
                        print(f"ℹ️ No text found in images")
                        status = "completed_image_only"
                else:
                    print(f"ℹ️ Image translation disabled - copying chapter as-is")
                    translated_html = c["body"]
                    status = "completed_image_only"
                
                safe_title = make_safe_filename(c['title'], c['num'])
                if isinstance(c['num'], float):
                    fname = f"response_{c['num']:06.1f}_{safe_title}.html"
                else:
                    fname = f"response_{c['num']:03d}_{safe_title}.html"
                
                with open(os.path.join(out, fname), 'w', encoding='utf-8') as f:
                    f.write(translated_html)
                
                print(f"[Chapter {idx+1}/{total_chapters}] ✅ Saved image-only chapter")
                progress_manager.update(idx, chap_num, content_hash, fname, status=status)
                progress_manager.save()
                chapters_completed += 1
                continue

            else:
                if is_mixed_content and image_translator and config.ENABLE_IMAGE_TRANSLATION:
                    print(f"🖼️ Processing {c.get('image_count', 0)} images first...")
                    
                    print(f"[DEBUG] Content before image processing (first 200 chars):")
                    print(c["body"][:200])
                    print(f"[DEBUG] Has h1 tags: {'<h1>' in c['body']}")
                    print(f"[DEBUG] Has h2 tags: {'<h2>' in c['body']}")
                    image_translator.set_current_chapter(chap_num)
                    c["body"], image_translations = process_chapter_images(
                        c["body"], 
                        actual_num,
                        image_translator,
                        check_stop
                    )
                    print(f"[DEBUG] Content after image processing (first 200 chars):")
                    print(c["body"][:200])
                    print(f"[DEBUG] Still has h1 tags: {'<h1>' in c['body']}")
                    print(f"[DEBUG] Still has h2 tags: {'<h2>' in c['body']}")
                    
                    if image_translations:
                        print(f"✅ Translated {len(image_translations)} images")
                    else:
                        print(f"ℹ️ No translatable text found in images")

                print(f"📖 Translating text content ({text_size} characters)")
                progress_manager.update(idx, chap_num, content_hash, output_filename=None, status="in_progress")
                progress_manager.save()
                
                _tok_env = os.getenv("MAX_INPUT_TOKENS", "1000000").strip()
                max_tokens_limit, budget_str = parse_token_limit(_tok_env)
                
                system_tokens = chapter_splitter.count_tokens(system)
                history_tokens = config.HIST_LIMIT * 2 * 1000
                safety_margin = 1000
                
                if max_tokens_limit is not None:
                    available_tokens = max_tokens_limit - system_tokens - history_tokens - safety_margin
                    chunks = chapter_splitter.split_chapter(c["body"], available_tokens)
                else:
                    chunks = [(c["body"], 1, 1)]
                
                print(f"📄 Chapter will be processed in {len(chunks)} chunk(s)")
            
            if len(chunks) > 1:
                chapter_tokens = chapter_splitter.count_tokens(c["body"])
                print(f"   ℹ️ Chapter size: {chapter_tokens:,} tokens (limit: {available_tokens:,} tokens per chunk)")
            else:
                chapter_tokens = chapter_splitter.count_tokens(c["body"])
                if max_tokens_limit is not None:
                    print(f"   ℹ️ Chapter size: {chapter_tokens:,} tokens (within limit of {available_tokens:,} tokens)")
            
            chapter_key_str = str(idx)
            if chapter_key_str not in progress_manager.prog["chapter_chunks"]:
                progress_manager.prog["chapter_chunks"][chapter_key_str] = {
                    "total": len(chunks),
                    "completed": [],
                    "chunks": {}
                }
            
            progress_manager.prog["chapter_chunks"][chapter_key_str]["total"] = len(chunks)
            
            translated_chunks = []
            
            for chunk_html, chunk_idx, total_chunks in chunks:
                chapter_key_str = content_hash
                old_key_str = str(idx)
                
                if chapter_key_str not in progress_manager.prog.get("chapter_chunks", {}) and old_key_str in progress_manager.prog.get("chapter_chunks", {}):
                    progress_manager.prog["chapter_chunks"][chapter_key_str] = progress_manager.prog["chapter_chunks"][old_key_str]
                    del progress_manager.prog["chapter_chunks"][old_key_str]
                    print(f"[PROGRESS] Migrated chunks for chapter {chap_num} to new tracking system")
                
                if chapter_key_str not in progress_manager.prog["chapter_chunks"]:
                    progress_manager.prog["chapter_chunks"][chapter_key_str] = {
                        "total": len(chunks),
                        "completed": [],
                        "chunks": {}
                    }
                
                progress_manager.prog["chapter_chunks"][chapter_key_str]["total"] = len(chunks)
                
                if chunk_idx in progress_manager.prog["chapter_chunks"][chapter_key_str]["completed"]:
                    saved_chunk = progress_manager.prog["chapter_chunks"][chapter_key_str]["chunks"].get(str(chunk_idx))
                    if saved_chunk:
                        translated_chunks.append((saved_chunk, chunk_idx, total_chunks))
                        print(f"  [SKIP] Chunk {chunk_idx}/{total_chunks} already translated")
                        continue
                        
                if config.CONTEXTUAL and history_manager.will_reset_on_next_append(config.HIST_LIMIT):
                    print(f"  📌 History will reset after this chunk (current: {len(history_manager.load_history())//2}/{config.HIST_LIMIT} exchanges)")
                    
                if check_stop():
                    print(f"❌ Translation stopped during chapter {idx+1}, chunk {chunk_idx}")
                    return
                
                current_chunk_number += 1
                
                progress_percent = (current_chunk_number / total_chunks_needed) * 100 if total_chunks_needed > 0 else 0
                
                if chunks_completed > 0:
                    elapsed_time = time.time() - translation_start_time
                    avg_time_per_chunk = elapsed_time / chunks_completed
                    remaining_chunks = total_chunks_needed - current_chunk_number + 1
                    eta_seconds = remaining_chunks * avg_time_per_chunk
                    
                    eta_hours = int(eta_seconds // 3600)
                    eta_minutes = int((eta_seconds % 3600) // 60)
                    eta_str = f"{eta_hours}h {eta_minutes}m" if eta_hours > 0 else f"{eta_minutes}m"
                else:
                    eta_str = "calculating..."
                
                if total_chunks > 1:
                    print(f"  🔄 Translating chunk {chunk_idx}/{total_chunks} for #{idx+1} (Overall: {current_chunk_number}/{total_chunks_needed} - {progress_percent:.1f}% - ETA: {eta_str})")
                    print(f"  ⏳ Chunk size: {len(chunk_html):,} characters (~{chapter_splitter.count_tokens(chunk_html):,} tokens)")
                else:
                    print(f"  📄 Translating chapter content (Overall: {current_chunk_number}/{total_chunks_needed} - {progress_percent:.1f}% - ETA: {eta_str}) [File: {c.get('original_basename', f'Chapter_{actual_num}')}]")
                    print(f"  📊 Chapter {actual_num} size: {len(chunk_html):,} characters (~{chapter_splitter.count_tokens(chunk_html):,} tokens)")
                
                print(f"  ℹ️ This may take 30-60 seconds. Stop will take effect after completion.")
                
                if log_callback:
                    if hasattr(log_callback, '__self__') and hasattr(log_callback.__self__, 'append_chunk_progress'):
                        if total_chunks == 1:
                            log_callback.__self__.append_chunk_progress(
                                1, 1, "text", 
                                f"Chapter {actual_num}",
                                overall_current=current_chunk_number,
                                overall_total=total_chunks_needed,
                                extra_info=f"{len(chunk_html):,} chars"
                            )
                        else:
                            log_callback.__self__.append_chunk_progress(
                                chunk_idx, 
                                total_chunks, 
                                "text", 
                                f"Chapter {actual_num}",
                                overall_current=current_chunk_number,
                                overall_total=total_chunks_needed
                            )
                    else:
                        if total_chunks == 1:
                            log_callback(f"📄 Processing Chapter {actual_num} ({chapters_completed + 1}/{chapters_to_process}) - {progress_percent:.1f}% complete")
                        else:
                            log_callback(f"📄 processing chunk {chunk_idx}/{total_chunks} for chapter {actual_num} - {progress_percent:.1f}% complete")   
                        
                if total_chunks > 1:
                    user_prompt = f"[PART {chunk_idx}/{total_chunks}]\n{chunk_html}"
                else:
                    user_prompt = chunk_html
                
                history = history_manager.load_history()
                
                if config.CONTEXTUAL:
                    trimmed = history[-config.HIST_LIMIT*2:]
                    
                    chunk_context = chunk_context_manager.get_context_messages(limit=2)
                    
                else:
                    trimmed = []
                    chunk_context = []

                if base_msg:
                    if not config.USE_ROLLING_SUMMARY:
                        filtered_base = [msg for msg in base_msg if "summary of the previous" not in msg.get("content", "")]
                        msgs = filtered_base + chunk_context + trimmed + [{"role": "user", "content": user_prompt}]
                    else:
                        msgs = base_msg + chunk_context + trimmed + [{"role": "user", "content": user_prompt}]
                else:
                    if trimmed or chunk_context:
                        msgs = chunk_context + trimmed + [{"role": "user", "content": user_prompt}]
                    else:
                        msgs = [{"role": "user", "content": user_prompt}]

                c['__index'] = idx
                c['__progress'] = progress_manager.prog
                c['history_manager'] = history_manager
                
                result, finish_reason = translation_processor.translate_with_retry(
                    msgs, chunk_html, c, chunk_idx, total_chunks
                )
                
                if result is None:
                    progress_manager.update(idx, actual_num, content_hash, output_filename=None, status="failed")
                    progress_manager.save()
                    continue

                if config.REMOVE_AI_ARTIFACTS:
                    result = ContentProcessor.clean_ai_artifacts(result, True)

                if config.EMERGENCY_RESTORE:
                    result = ContentProcessor.emergency_restore_paragraphs(result, chunk_html)

                if config.REMOVE_AI_ARTIFACTS:
                    lines = result.split('\n')
                    
                    json_line_count = 0
                    for i, line in enumerate(lines[:5]):
                        if line.strip() and any(pattern in line for pattern in [
                            '"role":', '"content":', '"messages":', 
                            '{"role"', '{"content"', '[{', '}]'
                        ]):
                            json_line_count = i + 1
                        else:
                            break
                    
                    if json_line_count > 0 and json_line_count < len(lines):
                        remaining = '\n'.join(lines[json_line_count:])
                        if remaining.strip() and len(remaining) > 100:
                            result = remaining
                            print(f"✂️ Removed {json_line_count} lines of JSON artifacts")

                result = re.sub(r'\[PART \d+/\d+\]\s*', '', result, flags=re.IGNORECASE)

                translated_chunks.append((result, chunk_idx, total_chunks))
                
                chunk_context_manager.add_chunk(user_prompt, result, chunk_idx, total_chunks)

                progress_manager.prog["chapter_chunks"][chapter_key_str]["completed"].append(chunk_idx)
                progress_manager.prog["chapter_chunks"][chapter_key_str]["chunks"][str(chunk_idx)] = result
                progress_manager.save()

                chunks_completed += 1
                    
                will_reset = history_manager.will_reset_on_next_append(
                    config.HIST_LIMIT if config.CONTEXTUAL else 0, 
                    config.TRANSLATION_HISTORY_ROLLING
                )

                if will_reset and config.USE_ROLLING_SUMMARY and config.CONTEXTUAL:
                    if check_stop():
                        print(f"❌ Translation stopped during summary generation for chapter {idx+1}")
                        return
                    
                    summary_resp = translation_processor.generate_rolling_summary(
                        history_manager, idx, chunk_idx
                    )
                    
                    if summary_resp:
                        base_msg[:] = [msg for msg in base_msg if "[MEMORY]" not in msg.get("content", "")]
                        
                        summary_msg = {
                            "role": os.getenv("SUMMARY_ROLE", "user"),
                            "content": (
                                "CONTEXT ONLY - DO NOT INCLUDE IN TRANSLATION:\n"
                                "[MEMORY] Previous context summary:\n\n"
                                f"{summary_resp}\n\n"
                                "[END MEMORY]\n"
                                "END OF CONTEXT - BEGIN ACTUAL CONTENT TO TRANSLATE:"
                            )
                        }
                        
                        if base_msg and base_msg[0].get("role") == "system":
                            base_msg.insert(1, summary_msg)
                        else:
                            base_msg.insert(0, summary_msg)

                history = history_manager.append_to_history(
                    user_prompt, 
                    result, 
                    config.HIST_LIMIT if config.CONTEXTUAL else 0,
                    reset_on_limit=True,
                    rolling_window=config.TRANSLATION_HISTORY_ROLLING
                )

                if chunk_idx < total_chunks:
                    for i in range(config.DELAY):
                        if check_stop():
                            print("❌ Translation stopped during delay")
                            return
                        time.sleep(1)

            if check_stop():
                print(f"❌ Translation stopped before saving chapter {idx+1}")
                return

            if len(translated_chunks) > 1:
                print(f"  📎 Merging {len(translated_chunks)} chunks...")
                translated_chunks.sort(key=lambda x: x[1])
                merged_result = chapter_splitter.merge_translated_chunks(translated_chunks)
            else:
                merged_result = translated_chunks[0][0] if translated_chunks else ""

            if config.CONTEXTUAL and len(translated_chunks) > 1:
                user_summary, assistant_summary = chunk_context_manager.get_summary_for_history()
                
                if user_summary and assistant_summary:
                    history_manager.append_to_history(
                        user_summary,
                        assistant_summary,
                        config.HIST_LIMIT,
                        reset_on_limit=False,
                        rolling_window=config.TRANSLATION_HISTORY_ROLLING
                    )
                    print(f"  📝 Added chapter summary to history")

            chunk_context_manager.clear()

            fname = FileUtilities.create_chapter_filename(c, actual_num)

            cleaned = re.sub(r"^```(?:html)?\s*\n?", "", merged_result, count=1, flags=re.MULTILINE)
            cleaned = re.sub(r"\n?```\s*$", "", cleaned, count=1, flags=re.MULTILINE)

            cleaned = ContentProcessor.clean_ai_artifacts(cleaned, remove_artifacts=config.REMOVE_AI_ARTIFACTS)

            with open(os.path.join(out, fname), 'w', encoding='utf-8') as f:
                f.write(cleaned)
            
            final_title = c['title'] or make_safe_filename(c['title'], actual_num)
            print(f"[Processed {idx+1}/{total_chapters}] ✅ Saved Chapter {actual_num}: {final_title}")
            
            progress_manager.update(idx, actual_num, content_hash, fname, status="completed")
            progress_manager.save()
            
            chapters_completed += 1

    if is_text_file:
        print("📄 Building final text file…")
        try:
            translated_files = []
            for chapter in chapters:
                fname = f"response_{chapter['num']:03d}_{make_safe_filename(chapter['title'], chapter['num'])}.html"
                if os.path.exists(os.path.join(out, fname)):
                    with open(os.path.join(out, fname), 'r', encoding='utf-8') as f:
                        content = f.read()
                    translated_files.append((fname, content))
            
            output_path = txt_processor.create_output_structure(translated_files)
            print(f"✅ Translation complete! Output saved to: {output_path}")
            
            total_time = time.time() - translation_start_time
            hours = int(total_time // 3600)
            minutes = int((total_time % 3600) // 60)
            seconds = int(total_time % 60)
            
            print(f"\n📊 Translation Statistics:")
            print(f"   • Total chunks processed: {chunks_completed}")
            print(f"   • Total time: {hours}h {minutes}m {seconds}s")
            
        except Exception as e:
            print("❌ Text file output build failed:", e)
    else:
        print("🔍 Checking for translated chapters...")
        response_files = [f for f in os.listdir(out) if f.startswith('response_') and f.endswith('.html')]
        chapter_files = [f for f in os.listdir(out) if f.startswith('chapter_') and f.endswith('.html')]

        if not response_files and chapter_files:
            print(f"⚠️ No translated files found, but {len(chapter_files)} original chapters exist")
            print("📝 Creating placeholder response files for EPUB compilation...")
            
            for chapter_file in chapter_files:
                response_file = chapter_file.replace('chapter_', 'response_', 1)
                src = os.path.join(out, chapter_file)
                dst = os.path.join(out, response_file)
                
                try:
                    with open(src, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    soup = BeautifulSoup(content, 'html.parser')
                    notice = soup.new_tag('p')
                    notice.string = "[Note: This chapter could not be translated - showing original content]"
                    notice['style'] = "color: red; font-style: italic;"
                    
                    if soup.body:
                        soup.body.insert(0, notice)
                    
                    with open(dst, 'w', encoding='utf-8') as f:
                        f.write(str(soup))
                        
                except Exception as e:
                    print(f"⚠️ Error processing {chapter_file}: {e}")
                    try:
                        shutil.copy2(src, dst)
                    except:
                        pass
            
            print(f"✅ Created {len(chapter_files)} placeholder response files")
            print("⚠️ Note: The EPUB will contain untranslated content")
        
        print("📘 Building final EPUB…")
        try:
            from epub_converter import fallback_compile_epub
            fallback_compile_epub(out, log_callback=log_callback)
            print("✅ All done: your final EPUB is in", out)
            
            total_time = time.time() - translation_start_time
            hours = int(total_time // 3600)
            minutes = int((total_time % 3600) // 60)
            seconds = int(total_time % 60)
            
            print(f"\n📊 Translation Statistics:")
            print(f"   • Total chunks processed: {chunks_completed}")
            print(f"   • Total time: {hours}h {minutes}m {seconds}s")
            if chunks_completed > 0:
                avg_time = total_time / chunks_completed
                print(f"   • Average time per chunk: {avg_time:.1f} seconds")
            
            stats = progress_manager.get_stats(out)
            print(f"\n📊 Progress Tracking Summary:")
            print(f"   • Total chapters tracked: {stats['total_tracked']}")
            print(f"   • Successfully completed: {stats['completed']}")
            print(f"   • Missing files: {stats['missing_files']}")
            print(f"   • In progress: {stats['in_progress']}")
                
        except Exception as e:
            print("❌ EPUB build failed:", e)

    print("TRANSLATION_COMPLETE_SIGNAL")

if __name__ == "__main__":
    main()
