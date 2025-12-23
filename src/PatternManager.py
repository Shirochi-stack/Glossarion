# -*- coding: utf-8 -*-
# =====================================================
# UNIFIED PATTERNS AND CONSTANTS
# Module-level constants for better pickle compatibility
# =====================================================
import re
import os
from bs4 import BeautifulSoup

# Move all patterns to module level for ProcessPoolExecutor compatibility
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
        # IMPORTANT: More specific patterns MUST come first
    r'^\d{3}(\d)_(\d{2})_\.x?html?$', # Captures both parts for decimal: group1.group2
    r'^\d{4}_(\d+)\.x?html?$',  # "0000_1.xhtml" - extracts 1, not 0000
    r'^\d+_(\d+)[_\.]',         # Any digits followed by underscore then capture next digits
    r'^(\d+)[_\.]',             # Standard: "0249_" or "0249."
    r'response_(\d+)_',         # Standard pattern: response_001_
    r'response_(\d+)\.',        # Pattern: response_001.
    r'(\d{3,5})[_\.]',          # 3-5 digit pattern with padding
    r'[Cc]hapter[_\s]*(\d+)',   # Chapter word pattern
    r'[Cc]h[_\s]*(\d+)',        # Ch abbreviation
    r'No(\d+)Chapter',          # No prefix with Chapter - matches "No00013Chapter.xhtml"
    r'No(\d+)Section',          # No prefix with Section - matches "No00013Section.xhtml"
    r'No(\d+)(?=\.|_|$)',       # No prefix followed by end, dot, or underscore (not followed by text)
    r'第(\d+)[章话回]',          # Chinese chapter markers
    r'_(\d+)(?:_|\.|$)',        # Number between underscores or at end
    r'^(\d+)(?:_|\.|$)',        # Starting with number
    r'(\d+)',                   # Any number (fallback)
]

CJK_HONORIFICS = {
    'korean': [
            # Modern honorifics
        '님', '씨', '선배', '후배', '동기', '형', '누나', '언니', '오빠', '동생',
        '선생님', '교수님', '박사님', '사장님', '회장님', '부장님', '과장님', '대리님',
        '팀장님', '실장님', '이사님', '전무님', '상무님', '부사장님', '고문님',
        '대표님', '원장님', '국장님', '차장님', '주임님', '반장님',
        '작가님', '기자님', '피디님', '감독님', '매니저님', '코치님',
        
            # Classical/formal honorifics
        '공', '옹', '군', '양', '낭', '랑', '생', '자', '부', '모', '시', '제', '족하',
        
            # Royal/noble address forms
        '마마', '마노라', '대감', '영감', '나리', '도령', '낭자', '아씨', '규수',
        '각하', '전하', '폐하', '저하', '합하', '대비', '대왕', '왕자', '공주',
        '빈궁', '중전', '세자', '군주', '태자', '성군',
        
            # Buddhist/religious
        '스님', '사부님', '조사님', '큰스님', '화상', '대덕', '대사', '법사',
        '선사', '율사', '보살님', '거사님', '신부님', '목사님', '장로님', '집사님',
        '전도사님', '수녀님', '교황님', '주교님',
        
            # Confucian/scholarly
        '부자', '선생', '대인', '어른', '어르신', '존자', '현자', '군자', '대부',
        '학사', '진사', '문하생', '제자', '유생', '선비',
        
            # Kinship honorifics
        '어르신', '할아버님', '할머님', '아버님', '어머님', '형님', '누님',
        '아주버님', '아주머님', '삼촌', '이모님', '고모님', '외삼촌', '장인어른',
        '장모님', '시아버님', '시어머님', '처남', '처형', '매형', '손님',
        '사돈', '백부님', '숙부님',
        
            # Verb-based honorific endings and speech levels (expanded)
        '습니다', 'ㅂ니다', '습니까', 'ㅂ니까', '시다', '세요', '셔요', '십시오', '시오',
        '이에요', '예요', '이예요', '에요', '어요', '아요', '여요', '해요', '이세요', '으세요',
        '으시', '시', '으십니다', '십니다', '으십니까', '십니까', '으셨', '셨',
        '드립니다', '드려요', '드릴게요', '드리겠습니다', '올립니다', '올려요',
        '사옵니다', '사뢰', '여쭙니다', '여쭤요', '아뢰', '뵙니다', '뵈요', '모십니다',
        '시지요', '시죠', '시네요', '시는군요', '시는구나', '으실', '실',
        '드시다', '잡수시다', '주무시다', '계시다', '가시다', '오시다',
        
            # Common verb endings with 있다/없다/하다
        '있어요', '있습니다', '있으세요', '있으십니까', '없어요', '없습니다', '없으세요',
        '해요', '합니다', '하세요', '하십시오', '하시죠', '하시네요', '했어요', '했습니다',
        '되세요', '되셨어요', '되십니다', '됩니다', '되요', '돼요',
        '이야', '이네', '이구나', '이군', '이네요', '인가요', '인가', '일까요', '일까',
        '거예요', '거에요', '겁니다', '건가요', '게요', '을게요', '을까요', '었어요', '었습니다',
        '겠습니다', '겠어요', '겠네요', '을겁니다', '을거예요', '을거에요',
        
            # Common endings
        '요', '죠', '네요', '는데요', '거든요', '니까', '으니까', '는걸요', '군요', '구나',
        '는구나', '는군요', '더라고요', '더군요', '던데요', '나요', '가요', '까요',
        '라고요', '다고요', '냐고요', '자고요', '란다', '단다', '냔다', '잔다',
        
            # Formal archaic endings
        '나이다', '사옵나이다', '옵니다', '오', '소서', '으오', '으옵소서', '사이다',
        '으시옵니다', '시옵니다', '으시옵니까', '시옵니까', '나이까', '리이까', '리이다',
        '옵소서', '으소서', '소이다', '로소이다', '이옵니다', '이올시다', '하옵니다'
    ],
    'japanese': [
            # Modern honorifics
        'さん', 'ちゃん', '君', 'くん', '様', 'さま', '先生', 'せんせい', '殿', 'どの', '先輩', 'せんぱい',
            # Classical/historical
        '氏', 'し', '朝臣', 'あそん', '宿禰', 'すくね', '連', 'むらじ', '臣', 'おみ', '君', 'きみ',
        '真人', 'まひと', '道師', 'みちのし', '稲置', 'いなぎ', '直', 'あたい', '造', 'みやつこ',
            # Court titles
        '卿', 'きょう', '大夫', 'たいふ', '郎', 'ろう', '史', 'し', '主典', 'さかん',
            # Buddhist titles
        '和尚', 'おしょう', '禅師', 'ぜんじ', '上人', 'しょうにん', '聖人', 'しょうにん',
        '法師', 'ほうし', '阿闍梨', 'あじゃり', '大和尚', 'だいおしょう',
            # Shinto titles
        '大宮司', 'だいぐうじ', '宮司', 'ぐうじ', '禰宜', 'ねぎ', '祝', 'はふり',
            # Samurai era
        '守', 'かみ', '介', 'すけ', '掾', 'じょう', '目', 'さかん', '丞', 'じょう',
            # Keigo (honorific language) verb forms
        'です', 'ます', 'ございます', 'いらっしゃる', 'いらっしゃいます', 'おっしゃる', 'おっしゃいます',
        'なさる', 'なさいます', 'くださる', 'くださいます', 'いただく', 'いただきます',
        'おります', 'でございます', 'ございません', 'いたします', 'いたしました',
        '申す', '申します', '申し上げる', '申し上げます', '存じる', '存じます', '存じ上げる',
        '伺う', '伺います', '参る', '参ります', 'お目にかかる', 'お目にかかります',
        '拝見', '拝見します', '拝聴', '拝聴します', '承る', '承ります',
            # Respectful prefixes/suffixes
        'お', 'ご', '御', 'み', '美', '貴', '尊'
    ],
    'chinese': [
            # Modern forms
        '先生', '小姐', '夫人', '公子', '大人', '老师', '师父', '师傅', '同志', '同学',
            # Ancient/classical forms
        '子', '丈', '翁', '公', '侯', '伯', '叔', '仲', '季', '父', '甫', '卿', '君', '生',
            # Imperial court
        '陛下', '殿下', '千岁', '万岁', '圣上', '皇上', '天子', '至尊', '御前', '爷',
            # Nobility/officials
        '阁下', '大人', '老爷', '相公', '官人', '郎君', '娘子', '夫子', '足下',
            # Religious titles
        '上人', '法师', '禅师', '大师', '高僧', '圣僧', '神僧', '活佛', '仁波切',
        '真人', '天师', '道长', '道友', '仙长', '上仙', '祖师', '掌教',
            # Scholarly/Confucian
        '夫子', '圣人', '贤人', '君子', '大儒', '鸿儒', '宗师', '泰斗', '巨擘',
            # Martial arts
        '侠士', '大侠', '少侠', '女侠', '英雄', '豪杰', '壮士', '义士',
            # Family/kinship
        '令尊', '令堂', '令郎', '令爱', '贤弟', '贤侄', '愚兄', '小弟', '家父', '家母',
            # Humble forms
        '在下', '小人', '鄙人', '不才', '愚', '某', '仆', '妾', '奴', '婢',
            # Polite verbal markers
        '请', '请问', '敢问', '恭请', '敬请', '烦请', '有请', '请教', '赐教',
        '惠顾', '惠赐', '惠存', '笑纳', '雅正', '指正', '斧正', '垂询',
        '拜', '拜见', '拜访', '拜读', '拜托', '拜谢', '敬上', '谨上', '顿首'
    ],
    'english': [
            # Modern Korean romanizations (Revised Romanization of Korean - 2000)
        '-nim', '-ssi', '-seonbae', '-hubae', '-donggi', '-hyeong', '-nuna', 
        '-eonni', '-oppa', '-dongsaeng', '-seonsaengnim', '-gyosunim', 
        '-baksanim', '-sajangnim', '-hoejangnim', '-bujangnim', '-gwajangnim',
        '-daerim', '-timjangnim', '-siljangnim', '-isanim', '-jeonmunim',
        '-sangmunim', '-busajangnim', '-gomunnim',
        
            # Classical/formal Korean romanizations  
        '-gong', '-ong', '-gun', '-yang', '-nang', '-rang', '-saeng', '-ja',
        '-bu', '-mo', '-si', '-je', '-jokha',
        
            # Royal/noble Korean romanizations
        '-mama', '-manora', '-daegam', '-yeonggam', '-nari', '-doryeong',
        '-nangja', '-assi', '-gyusu', '-gakha', '-jeonha', '-pyeha', '-jeoha',
        '-hapka', '-daebi', '-daewang', '-wangja', '-gongju',
        
            # Buddhist/religious Korean romanizations
        '-seunim', '-sabunim', '-josanim', '-keunseunim', '-hwasang',
        '-daedeok', '-daesa', '-beopsa', '-seonsa', '-yulsa', '-bosalnim',
        '-geosanim', '-sinbunim', '-moksanim', '-jangnonim', '-jipsanim',
        
            # Confucian/scholarly Korean romanizations
        '-buja', '-seonsaeng', '-daein', '-eoreun', '-eoreusin', '-jonja', 
        '-hyeonja', '-gunja', '-daebu', '-haksa', '-jinsa', '-munhasaeng', '-jeja',
        
            # Kinship Korean romanizations
        '-harabeonim', '-halmeonim', '-abeonim', '-eomeonim', '-hyeongnim', 
        '-nunim', '-ajubeonim', '-ajumeonim', '-samchon', '-imonim', '-gomonim',
        '-oesamchon', '-jangineoreun', '-jangmonim', '-siabeonim', '-sieomeonim',
        '-cheonam', '-cheohyeong', '-maehyeong', '-sonnim',
        
            # Korean verb endings romanized (Revised Romanization)
        '-seumnida', '-mnida', '-seumnikka', '-mnikka', '-sida', '-seyo', 
        '-syeoyo', '-sipsio', '-sio', '-ieyo', '-yeyo', '-iyeyo', '-eyo', 
        '-eoyo', '-ayo', '-yeoyo', '-haeyo', '-iseyo', '-euseyo',
        '-eusi', '-si', '-eusimnida', '-simnida', '-eusimnikka', '-simnikka',
        '-eusyeot', '-syeot', '-deurimnida', '-deuryeoyo', '-deurilgeyo',
        '-deurigesseumnida', '-ollimnida', '-ollyeoyo', '-saomnida', '-saroe',
        '-yeojjumnida', '-yeojjwoyo', '-aroe', '-boemnida', '-boeyo', '-mosimnida',
        '-sijiyo', '-sijyo', '-sineyo', '-sineungunyo', '-sineunguna', '-eusil', '-sil',
        '-deusida', '-japsusida', '-jumusida', '-gyesida', '-gasida', '-osida',
        
            # Common Korean verb endings romanized
        '-isseoyo', '-isseumnida', '-isseuseyo', '-isseusimnikka', 
        '-eopseoyo', '-eopseumnida', '-eopseuseyo', '-hamnida', '-haseyo', 
        '-hasipsio', '-hasijyo', '-hasineyo', '-haesseoyo', '-haesseumnida',
        '-doeseyo', '-doesyeosseoyo', '-doesimnida', '-doemnida', '-doeyo', '-dwaeyo',
        '-iya', '-ine', '-iguna', '-igun', '-ineyo', '-ingayo', '-inga', 
        '-ilkkayo', '-ilkka', '-geoyeyo', '-geoeyo', '-geomnida', '-geongayo',
        '-geyo', '-eulgeyo', '-eulkkayo', '-eosseoyo', '-eosseumnida',
        '-gesseumnida', '-gesseoyo', '-genneyo', '-eulgeommida', '-eulgeoyeyo', '-eulgeoeyo',
        
            # Common Korean endings romanized
        '-yo', '-jyo', '-neyo', '-neundeyo', '-geodeunyo', '-nikka', 
        '-eunikka', '-neungeolyo', '-gunyo', '-guna', '-neunguna', '-neungunyo',
        '-deoragoyo', '-deogunyo', '-deondeyo', '-nayo', '-gayo', '-kkayo',
        '-ragoyo', '-dagoyo', '-nyagoyo', '-jagoyo', '-randa', '-danda', 
        '-nyanda', '-janda',
        
            # Formal archaic Korean romanized
        '-naida', '-saomnaida', '-omnida', '-o', '-soseo', '-euo', 
        '-euopsoseo', '-saida', '-eusiomnida', '-siomnida', '-eusiomnikka', 
        '-siomnikka', '-naikka', '-riikka', '-riida', '-opsoseo', '-eusoseo',
        '-soida', '-rosoida', '-iomnida', '-iolsida', '-haomnida',
        
            # Japanese keigo romanized (keeping existing)
        '-san', '-chan', '-kun', '-sama', '-sensei', '-senpai', '-dono', 
        '-shi', '-tan', '-chin', '-desu', '-masu', '-gozaimasu', 
        '-irassharu', '-irasshaimasu', '-ossharu', '-osshaimasu',
        '-nasaru', '-nasaimasu', '-kudasaru', '-kudasaimasu', '-itadaku', 
        '-itadakimasu', '-orimasu', '-degozaimasu', '-gozaimasen', 
        '-itashimasu', '-itashimashita', '-mousu', '-moushimasu', 
        '-moushiageru', '-moushiagemasu', '-zonjiru', '-zonjimasu',
        '-ukagau', '-ukagaimasu', '-mairu', '-mairimasu', '-haiken', 
        '-haikenshimasu',
        
            # Chinese romanizations (keeping existing)
        '-xiong', '-di', '-ge', '-gege', '-didi', '-jie', '-jiejie', 
        '-meimei', '-shixiong', '-shidi', '-shijie', '-shimei', '-gongzi', 
        '-guniang', '-xiaojie', '-daren', '-qianbei', '-daoyou', '-zhanglao', 
        '-shibo', '-shishu', '-shifu', '-laoshi', '-xiansheng', '-daxia', 
        '-shaoxia', '-nvxia', '-jushi', '-shanren', '-dazhang', '-zhenren',
        'benzuo', 'bengong', 'benwang', 'benshao', 'zhen', 'gu', 'laozi', 'zaixia',
        'pindao', 'xiaodao', 'nucai', 'chen', 'qie', 'wanbei',
        'bixia', 'dianxia', 'niangniang', 'laoda', 'laoban', 'zhanggui', 'xiaoer',
        'shizhu',
        
            # Ancient Chinese romanizations
        '-zi', '-gong', '-hou', '-bo', '-jun', '-qing', '-weng', '-fu', 
        '-sheng', '-lang', '-langjun', '-niangzi', '-furen', '-gege', 
        '-jiejie', '-yeye', '-nainai',
        
            # Chinese politeness markers romanized
        '-qing', '-jing', '-gong', '-hui', '-ci', '-bai', '-gan', '-chui',
        'qingwen', 'ganwen', 'gongjing', 'jingjing', 'baijian', 'baifang', 
        'baituo'
    ]
}

TITLE_PATTERNS = {
    'korean': [
            # Modern titles
        r'\b(왕|여왕|왕자|공주|황제|황후|대왕|대공|공작|백작|자작|남작|기사|장군|대장|원수|제독|함장|대신|재상|총리|대통령|시장|지사|검사|판사|변호사|의사|박사|교수|신부|목사|스님|도사)\b',
        r'\b(폐하|전하|각하|예하|님|대감|영감|나리|도련님|아가씨|부인|선생)\b',
            # Historical/classical titles
        r'\b(대왕|태왕|왕비|왕후|세자|세자빈|대군|군|옹주|공주|부마|원자|원손)\b',
        r'\b(영의정|좌의정|우의정|판서|참판|참의|정승|판사|사또|현령|군수|목사|부사)\b',
        r'\b(대제학|제학|대사간|사간|대사헌|사헌|도승지|승지|한림|사관|내시|환관)\b',
        r'\b(병조판서|이조판서|호조판서|예조판서|형조판서|공조판서)\b',
        r'\b(도원수|부원수|병마절도사|수군절도사|첨절제사|만호|천호|백호)\b',
        r'\b(정일품|종일품|정이품|종이품|정삼품|종삼품|정사품|종사품|정오품|종오품)\b',
            # Korean honorific verb endings patterns
        r'(습니다|ㅂ니다|습니까|ㅂ니까|세요|셔요|십시오|시오)$',
        r'(이에요|예요|이예요|에요|어요|아요|여요|해요)$',
        r'(으시|시)(었|겠|ㄹ|을|는|던)*(습니다|ㅂ니다|어요|아요|세요)',
        r'(드립니다|드려요|드릴게요|드리겠습니다|올립니다|올려요)$',
        r'(사옵니다|여쭙니다|여쭤요|뵙니다|뵈요|모십니다)$',
        r'(나이다|사옵나이다|옵니다|으오|으옵소서|사이다)$'
    ],
    'japanese': [
            # Modern titles
        r'\b(王|女王|王子|姫|皇帝|皇后|天皇|皇太子|大王|大公|公爵|伯爵|子爵|男爵|騎士|将軍|大将|元帥|提督|艦長|大臣|宰相|総理|大統領|市長|知事|検事|裁判官|弁護士|医者|博士|教授|神父|牧師|僧侶|道士)\b',
        r'\b(陛下|殿下|閣下|猊下|様|大人|殿|卿|君|氏)\b',
            # Historical titles
        r'\b(天皇|皇后|皇太子|親王|内親王|王|女王|太政大臣|左大臣|右大臣|内大臣|大納言|中納言|参議)\b',
        r'\b(関白|摂政|征夷大将軍|管領|執権|守護|地頭|代官|奉行|与力|同心)\b',
        r'\b(太政官|神祇官|式部省|治部省|民部省|兵部省|刑部省|大蔵省|宮内省)\b',
        r'\b(大僧正|僧正|大僧都|僧都|律師|大法師|法師|大禅師|禅師)\b',
        r'\b(正一位|従一位|正二位|従二位|正三位|従三位|正四位|従四位|正五位|従五位)\b',
        r'\b(大和守|山城守|摂津守|河内守|和泉守|伊賀守|伊勢守|尾張守|三河守|遠江守)\b',
            # Japanese keigo (honorific language) patterns
        r'(です|ます|ございます)$',
        r'(いらっしゃ|おっしゃ|なさ|くださ)(います|いました|る|った)$',
        r'(いただ|お|ご|御)(き|きます|きました|く|ける|けます)',
        r'(申し上げ|申し|存じ上げ|存じ|伺い|参り)(ます|ました|る)$',
        r'(拝見|拝聴|承り|承)(します|しました|いたします|いたしました)$',
        r'お[^あ-ん]+[になる|になります|くださる|くださいます]'
    ],
    'chinese': [
            # Modern titles
        r'\b(王|女王|王子|公主|皇帝|皇后|大王|大公|公爵|伯爵|子爵|男爵|骑士|将军|大将|元帅|提督|舰长|大臣|宰相|总理|大总统|市长|知事|检察官|法官|律师|医生|博士|教授|神父|牧师|和尚|道士)\b',
        r'\b(陛下|殿下|阁下|大人|老爷|夫人|小姐|公子|少爷|姑娘|先生)\b',
            # Imperial titles
        r'\b(天子|圣上|皇上|万岁|万岁爷|太上皇|皇太后|太后|皇后|贵妃|妃|嫔|贵人|常在|答应)\b',
        r'\b(太子|皇子|皇孙|亲王|郡王|贝勒|贝子|公主|格格|郡主|县主|郡君|县君)\b',
            # Ancient official titles
        r'\b(丞相|相国|太师|太傅|太保|太尉|司徒|司空|大司马|大司农|大司寇)\b',
        r'\b(尚书|侍郎|郎中|员外郎|主事|知府|知州|知县|同知|通判|推官|巡抚|总督)\b',
        r'\b(御史大夫|御史中丞|监察御史|给事中|都察院|翰林院|国子监|钦天监)\b',
        r'\b(大学士|学士|侍读|侍讲|编修|检讨|庶吉士|举人|进士|状元|榜眼|探花)\b',
            # Military ranks
        r'\b(大元帅|元帅|大将军|将军|都督|都指挥使|指挥使|千户|百户|总兵|副将|参将|游击|都司|守备)\b',
        r'\b(提督|总兵官|副总兵|参将|游击将军|都司|守备|千总|把总|外委)\b',
            # Religious titles
        r'\b(国师|帝师|法王|活佛|堪布|仁波切|大和尚|方丈|住持|首座|维那|知客)\b',
        r'\b(天师|真人|道长|掌教|监院|高功|都讲|总理|提点|知观)\b',
            # Nobility ranks
        r'\b(公|侯|伯|子|男|开国公|郡公|国公|郡侯|县侯|郡伯|县伯|县子|县男)\b',
        r'\b(一品|二品|三品|四品|五品|六品|七品|八品|九品|正一品|从一品|正二品|从二品)\b',
            # Chinese politeness markers
        r'(请|敢|恭|敬|烦|有)(问|请|赐|教|告|示)',
        r'(拜|惠|赐|垂|雅|笑)(见|访|读|托|谢|顾|赐|存|纳|正|询)',
        r'(敬|谨|顿)(上|呈|启|白|首)'
    ],
    'english': [
            # Western titles
        r'\b(King|Queen|Prince|Princess|Emperor|Empress|Duke|Duchess|Marquis|Marquess|Earl|Count|Countess|Viscount|Viscountess|Baron|Baroness|Knight|Lord|Lady|Sir|Dame|General|Admiral|Captain|Major|Colonel|Commander|Lieutenant|Sergeant|Minister|Chancellor|President|Mayor|Governor|Judge|Doctor|Professor|Father|Reverend|Master|Mistress)\b',
        r'\b(His|Her|Your|Their)\s+(Majesty|Highness|Grace|Excellency|Honor|Worship|Lordship|Ladyship)\b',
            # Romanized historical titles
        r'\b(Tianzi|Huangdi|Huanghou|Taizi|Qinwang|Junwang|Beile|Beizi|Gongzhu|Gege|Bixia|Dianxia|Niangniang|Fuma|Wangye)\b',
        r'\b(Chengxiang|Zaixiang|Taishi|Taifu|Taibao|Taiwei|Situ|Sikong|Dasima)\b',
        r'\b(Shogun|Daimyo|Samurai|Ronin|Ninja|Tenno|Mikado|Kampaku|Sessho)\b',
        r'\b(Taewang|Wangbi|Wanghu|Seja|Daegun|Gun|Ongju|Gongju|Buma)\b'
    ]
}

    # Expanded Chinese numbers including classical forms
# Chinese compound surnames (two-character surnames)
CHINESE_COMPOUND_SURNAMES = {
    '司马', '欧阳', '上官', '诸葛', '慕容', '皇甫', '尉迟', '公孙',
    '轩辕', '令狐', '南宫', '东方', '西门', '独孤', '完颜', '赫连',
    '澹台', '公羊', '钟离', '长孙', '宇文', '百里', '呼延', '东郭',
    '南门', '羊舌', '微生', '梁丘', '左丘', '段干', '端木', '夏侯'
}

# Common single-character Chinese surnames
CHINESE_SINGLE_SURNAMES = {
    '赵', '钱', '孙', '李', '周', '吴', '郑', '王', '冯', '陈',
    '褚', '卫', '蒋', '沈', '韩', '杨', '朱', '秦', '尤', '许',
    '何', '吕', '施', '张', '孔', '曹', '严', '华', '金', '魏',
    '陶', '姜', '戚', '谢', '邹', '喻', '柏', '水', '窦', '章',
    '云', '苏', '潘', '葛', '奚', '范', '彭', '郎', '鲁', '韦',
    '昌', '马', '苗', '凤', '花', '方', '俞', '任', '袁', '柳',
    '酆', '鲍', '史', '唐', '费', '廉', '岑', '薛', '雷', '贺',
    '倪', '汤', '滕', '殷', '罗', '毕', '郝', '邬', '安', '常',
    '乐', '于', '时', '傅', '皮', '卞', '齐', '康', '伍', '余',
    '元', '卜', '顾', '孟', '平', '黄', '和', '穆', '萧', '尹',
    '姚', '邵', '湛', '汪', '祁', '毛', '禹', '狄', '米', '贝',
    '明', '臧', '计', '伏', '成', '戴', '谈', '宋', '茅', '庞',
    '熊', '纪', '舒', '屈', '项', '祝', '董', '梁', '杜', '阮',
    '蓝', '闵', '席', '季', '麻', '强', '贾', '路', '娄', '危'
}

# Chinese cultivation/xianxia terms (common in web novels)
CHINESE_CULTIVATION_TERMS = {
    'realms': [
        '练气', '筑基', '金丹', '元婴', '化神', '炼虚', '合体', '渡劫', '大乘',
        '凝气', '开光', '融合', '心动', '灵寂', '出窍', '分神', '反虚', '天劫',
        '先天', '后天', '武徒', '武者', '武师', '大武师', '武灵', '武王', '武皇', '武帝', '武圣',
        '斗之气', '斗者', '斗师', '大斗师', '斗灵', '斗王', '斗皇', '斗宗', '斗尊', '斗圣', '斗帝'
    ],
    'techniques': [
        '剑诀', '心法', '功法', '秘术', '神通', '法术', '仙术', '道法', '魔功',
        '剑法', '刀法', '掌法', '拳法', '指法', '腿法', '身法', '步法',
        '御剑术', '御剑飞行', '遁术', '遁法', '土遁', '火遁', '水遁'
    ],
    'items': [
        '法宝', '灵器', '仙器', '神器', '圣器', '道器', '魔器',
        '丹药', '灵丹', '仙丹', '神丹', '灵石', '灵晶', '仙石',
        '灵草', '灵药', '仙草', '天材地宝', '灵兽', '妖兽', '凶兽'
    ],
    'titles': [
        '真人', '道友', '师兄', '师姐', '师弟', '师妹', '掌教', '长老',
        '太上长老', '峰主', '殿主', '宗主', '教主', '盟主', '魔头',
        '散修', '剑修', '体修', '炼丹师', '炼器师', '阵法师', '符师'
    ],
    'locations': [
        '洞府', '洞天', '秘境', '小世界', '福地', '圣地', '禁地', '遗迹',
        '山门', '主峰', '灵峰', '药园', '藏经阁', '传功殿', '炼器阁'
    ]
}

# Chinese wuxia terms (martial arts novels)
CHINESE_WUXIA_TERMS = {
    'skills': [
        '轻功', '内功', '外功', '硬功', '软功', '气功', '真气', '内力',
        '降龙十八掌', '六脉神剑', '九阴真经', '九阳神功', '易筋经', '葵花宝典'
    ],
    'sects': [
        '少林', '武当', '峨眉', '华山', '恒山', '衡山', '嵩山', '泰山',
        '全真', '古墓', '桃花岛', '丐帮', '明教', '日月神教', '星宿',
        '门派', '教派', '帮派', '世家', '家族'
    ],
    'ranks': [
        '武林盟主', '一流高手', '二流高手', '三流高手', '绝顶高手',
        '宗师', '大宗师', '先天宗师', '大侠', '少侠', '女侠'
    ]
}

# Common Chinese terms found in web novels (Wuxia/Xianxia/Xuanhuan)
CHINESE_NOVEL_TERMS = {
    'cultivation': [
        '气', '丹田', '经脉', '真气', '元气', '灵气', '仙气', '玄气', '斗气', '魔气', '鬼气', '妖气', '煞气', '罡气', '剑气', '刀气',
        '金丹', '元婴', '筑基', '练气', '心魔', '识海', '神识', '元神', '灵魂', '肉身', '法身',
        '道', '阴阳', '五行', '武', '功', '法', '术', '神通', '道法', '仙术', '禁制', '阵法', '符箓',
        '劫', '天劫', '雷劫', '心魔劫', '飞升', '渡劫', '顿悟', '闭关', '出关', '夺舍', '转世', '轮回',
        '炼丹', '炼器', '炼阵', '炼符', '双修', '炉鼎', '采补', '辟谷'
    ],
    'beings': [
        '仙', '魔', '妖', '鬼', '神', '人', '龙', '凤', '麒麟', '玄武', '白虎', '朱雀',
        '尊者', '圣人', '大帝', '天尊', '道祖', '真人', '上仙', '散仙', '地仙', '天仙', '金仙',
        '散修', '邪修', '魔修', '鬼修', '妖修', '剑修', '体修', '法修', '器灵', '傀儡'
    ],
    'world': [
        '江湖', '武林', '天下', '宗门', '帮', '派', '家', '世家', '皇朝', '帝国', '圣地',
        '仙界', '神界', '魔界', '妖界', '鬼界', '凡间', '俗世', '修真界',
        '秘境', '遗迹', '洞府', '福地', '洞天', '禁地', '绝地', '坊市', '拍卖会'
    ],
    'address_self': [
        '本座', '本宫', '本王', '本少', '本皇', '本帝', '本尊', '本圣',
        '朕', '孤', '寡人', '老子', '老夫', '老身', '老朽', '老衲',
        '在下', '鄙人', '小道', '贫道', '贫僧', '小僧', '不才',
        '奴才', '臣', '妾', '妾身', '晚辈', '小弟', '小妹', '学生', '弟子', '徒儿'
    ],
    'address_others': [
        '陛下', '殿下', '娘娘', '王爷', '侯爷',
        '老大', '老板', '掌柜', '小二',
        '道友', '施主', '大师', '师太', '真人', '仙子', '仙长',
        '前辈', '后辈', '小友',
        '师尊', '师父', '师傅', '师兄', '师弟', '师姐', '师妹', '师叔', '师伯', '师祖',
        '兄台', '仁兄', '贤弟',
        '公子', '少爷', '小姐', '姑娘', '夫人', '老爷', '太太'
    ],
    'measurements': [
        '里', '丈', '尺', '寸',
        '斤', '两',
        '时辰', '刻', '分', '息', '炷香', '盏茶',
        '元', '文', '两', '贯'
    ],
    'items': [
        '法宝', '灵宝', '仙宝', '古宝', '灵器', '法器', '仙器', '神器',
        '丹药', '灵丹', '仙丹', '毒丹', '废丹',
        '灵石', '仙石', '灵晶', '玄石',
        '纳戒', '储物袋', '储物戒指', '乾坤袋',
        '玉简', '功法', '秘籍', '图谱',
        '灵草', '灵药', '仙草', '天材地宝'
    ]
}

# Chinese relationship and family terms (important for character relationships)
CHINESE_RELATIONSHIP_TERMS = {
    'family': [
        '父亲', '母亲', '爷爷', '奶奶', '外公', '外婆', '伯父', '伯母',
        '叔父', '叔母', '姑父', '姑母', '姨父', '姨母', '舅父', '舅母',
        '哥哥', '姐姐', '弟弟', '妹妹', '兄长', '兄弟', '姐妹',
        '儿子', '女儿', '孙子', '孙女', '外孙', '外孙女',
        '夫君', '夫人', '妻子', '夫妻', '娘子', '娘亲', '相公'
    ],
    'master_disciple': [
        '师父', '师尊', '师傅', '师娘', '师祖', '师父祖', '太师父',
        '徒弟', '徒儿', '师兄', '师姐', '师弟', '师妹', '师叔', '师伯',
        '师兄弟', '师姐妹', '同门', '师门', '亲传弟子', '内门弟子', '外门弟子'
    ],
    'sworn': [
        '义父', '义母', '义兄', '义弟', '义兄弟', '义子',
        '结义', '义结金兰', '殃血为盟', '拜把兄弟'
    ],
    'romantic': [
        '道侣', '伴侣', '知己', '红颜', '知己', '佳人', '爱人',
        '未婚妻', '未婚夫', '娘子', '如意郎君'
    ]
}

# Chinese mythological and historical elements
CHINESE_MYTHOLOGICAL_TERMS = {
    'creatures': [
        '龙', '凤凰', '麒麟', '玄武', '白虎', '朱雀', '青龙',
        '天马', '龙马', '神鹰', '凤凰', '火凤', '冰凤',
        '虎', '狼', '豹', '熊', '鹰', '蛇', '蛟', '蛟龙',
        '龙龟', '神兽', '圣兽', '妖兽', '魔兽', '凶兽', '荒兽'
    ],
    'divine_artifacts': [
        '上古神器', '先天灵宝', '先天至宝', '混沌至宝',
        '开天神斧', '盘古幡', '太极图', '混沌钟', '造化鼎',
        '九鼎', '十大神器', '上古十大凶剑'
    ],
    'heavenly': [
        '天庭', '天宫', '天界', '人间', '地界', '冥界', '修罗界', '阿修罗界',
        '九重天', '三十三天', '三十三重天', '九幽', '黄泉',
        '三界', '六道', '六道轮回', '八荒', '四海'
    ],
    'legendary': [
        '三皇', '五帝', '盘古', '女娲', '伏義', '神农', '烎帝', '黄帝',
        '仙人', '真仭', '大能', '大帝', '天尊', '圣人', '至尊', '道祖'
    ]
}

# Chinese elemental and natural forces
CHINESE_ELEMENTAL_TERMS = {
    'five_elements': [
        '金', '木', '水', '火', '土', '金属性', '木属性', '水属性', '火属性', '土属性',
        '金灵根', '木灵根', '水灵根', '火灵根', '土灵根',
        '五行', '五行之力', '五行相生', '五行相克'
    ],
    'yin_yang': [
        '阴', '阳', '阴阳', '阴阳之力', '阴阳二气', '阴阳调和',
        '至阴', '至阳', '纯阴', '纯阳', '太阴', '太阳',
        '阴气', '阳气', '阴寒', '阳炎'
    ],
    'natural_forces': [
        '风', '雷', '冰', '电', '光', '暗', '空间', '时间',
        '风属性', '雷属性', '冰属性', '电属性',
        '狂风', '雷霆', '冰霆', '火焰', '水流', '地震',
        '天地玄黄', '混沌之力', '鸿蒙之气', '先天之气'
    ]
}

# Chinese body cultivation and physique types
CHINESE_PHYSIQUE_TERMS = {
    'special_physiques': [
        '先天道体', '先天霸体', '先天圣体', '混沌体',
        '九阴体', '九阳体', '阴阳体', '五行体',
        '剑体', '剑骨', '刀骨', '剑心', '刀心',
        '霸体', '圣体', '魔体', '仙体', '神体',
        '无垢道体', '无漏金身', '金刚不坏', '不死之身'
    ],
    'spiritual_roots': [
        '灵根', '天灵根', '地灵根', '天资', '根骨',
        '单灵根', '双灵根', '三灵根', '四灵根', '五灵根',
        '异灵根', '变异灵根', '绝世灵根',
        '废灵根', '伪灵根', '双修', '全灵根'
    ]
}

# Chinese treasure and artifact grades
CHINESE_TREASURE_GRADES = {
    'grades': [
        '凡器', '凡品', '普通', '低阶', '中阶', '高阶',
        '灵器', '宝器', '法器', '道器', '仙器', '神器', '圣器',
        '下品', '中品', '上品', '极品', '绝品',
        '天阶', '地阶', '玄阶', '黄阶',
        '一品', '二品', '三品', '四品', '五品', '六品', '七品', '八品', '九品'
    ],
    'pill_grades': [
        '一纹', '二纹', '三纹', '四纹', '五纹', '六纹', '七纹', '八纹', '九纹',
        '一转', '二转', '三转', '四转', '五转', '六转', '七转', '八转', '九转',
        '丹云', '丹纹', '丹雷', '丹香'
    ]
}

# Chinese naming conventions and patterns
CHINESE_NAME_PATTERNS = {
    'courtesy_names': [
        # Pattern: Character + 字 (courtesy name marker)
        r'[\u4e00-\u9fff]{1,2}字[\u4e00-\u9fff]{1,2}',  # X字Y format
    ],
    'generation_names': [
        # Common generation name characters (used in family naming)
        '文', '武', '明', '德', '仁', '义', '礼', '智', '信',
        '宝', '珍', '玉', '金', '银', '富', '贵', '康', '宁',
        '光', '耀', '华', '荣', '昌', '盛', '兴', '隆'
    ],
    'title_prefixes': [
        # Common prefixes for titles and names in novels
        '老', '小', '大', '少', '诸',  # Old, Young, Great, Young Master, All
        '无', '玄', '火', '冰', '雷', '风', '天', '魔', '剑', '刀',  # Elemental prefixes
        '龙', '凤', '虎', '鹰', '狼', '豹',  # Animal prefixes
        '圣', '魔', '仙', '神', '鬼', '妖'  # Divine/demonic prefixes
    ],
    'clan_prefixes': [
        # Common clan/family name structures
        '家族', '世家', '氏族', '宗族', '一族',
        '皇族', '神族', '魔族', '妖族', '古族'
    ]
}

# Additional cultivation power systems (system novels, game novels)
CHINESE_POWER_SYSTEMS = {
    'levels': [
        '一级', '二级', '三级', '四级', '五级', '六级', '七级', '八级', '九级', '十级',
        '初级', '中级', '高级', '顶级', '工级',
        '一阶', '二阶', '三阶', '四阶', '五阶', '六阶', '七阶', '八阶', '九阶',
        '青铜', '白银', '黄金', '铂金', '钻石', '王者', '皇者'
    ],
    'stars_moons': [
        '一星', '二星', '三星', '四星', '五星', '六星', '七星', '八星', '九星',
        '一月', '二月', '三月', '四月', '五月', '六月', '七月', '八月', '九月',
        '一轮', '二轮', '三轮', '四轮', '五轮', '六轮', '七轮', '八轮', '九轮',
        '半月', '满月', '新月'
    ],
    'circles_rings': [
        '一环', '二环', '三环', '四环', '五环', '六环', '七环', '八环', '九环',
        '一圈', '二圈', '三圈', '四圈', '五圈', '六圈', '七圈', '八圈', '九圈',
        '魂环', '魂圈', '灵环', '灵圈'
    ],
    'colors': [
        '白色', '黄色', '紫色', '黑色', '红色', '蓝色', '绿色', '金色', '银色',
        '白级', '黄级', '紫级', '黑级', '红级', '蓝级', '绿级', '金级', '银级'
    ]
}

# Chinese location types (common in novels)
CHINESE_LOCATION_TYPES = {
    'buildings': [
        '宫', '殿', '阁', '堂', '馆', '院', '楼', '轩', '亭', '台',
        '大殿', '主殿', '左殿', '右殿', '偷殿', '后殿',
        '藏经阁', '传功阁', '炼器阁', '炼丹阁', '任务大厅'
    ],
    'natural': [
        '山', '峰', '岭', '水', '河', '湖', '海', '江', '谷', '林', '洞',
        '主峰', '北峰', '南峰', '东峰', '西峰',
        '大山', '神山', '魔山', '仙山', '灵山',
        '星海', '星空', '星域', '星球', '星辰'
    ],
    'regions': [
        '域', '境', '界', '地', '州', '郡', '城', '镇', '村', '坊',
        '东域', '西域', '南域', '北域', '中域',
        '上界', '中界', '下界', '下位界', '中位界', '上位界',
        '修真界', '仙界', '神界', '魔界', '佛界', '妖界'
    ]
}

# Chinese battle and technique descriptors
CHINESE_BATTLE_TERMS = {
    'attack_types': [
        '攻击', '防御', '身法', '步法', '速度', '力量',
        '剑气', '刀气', '拳劲', '掌力', '指力', '腿力',
        '真元', '魂力', '精神力', '神识', '神念'
    ],
    'techniques_suffixes': [
        '诀', '法', '术', '功', '心法', '秘法', '神通',
        '一式', '一招', '一击', '一指', '一掌', '一剑', '一刀'
    ],
    'power_descriptors': [
        '无敌', '无双', '无上', '至强', '至尊', '极致', '绝世',
        '霸道', '王道', '圣道', '仙道', '神道', '魔道'
    ]
}

CHINESE_NUMS = {
        # Basic numbers
    '一': 1, '二': 2, '三': 3, '四': 4, '五': 5,
    '六': 6, '七': 7, '八': 8, '九': 9, '十': 10,
    '十一': 11, '十二': 12, '十三': 13, '十四': 14, '十五': 15,
    '十六': 16, '十七': 17, '十八': 18, '十九': 19, '二十': 20,
    '二十一': 21, '二十二': 22, '二十三': 23, '二十四': 24, '二十五': 25,
    '三十': 30, '四十': 40, '五十': 50, '六十': 60,
    '七十': 70, '八十': 80, '九十': 90, '百': 100,
        # Classical/formal numbers
    '壹': 1, '贰': 2, '叁': 3, '肆': 4, '伍': 5,
    '陆': 6, '柒': 7, '捌': 8, '玖': 9, '拾': 10,
    '佰': 100, '仟': 1000, '萬': 10000, '万': 10000,
        # Ordinal indicators
    '第一': 1, '第二': 2, '第三': 3, '第四': 4, '第五': 5,
    '首': 1, '次': 2, '初': 1, '末': -1,
}

# Gender indicator patterns for pronoun-based gender detection
# Note: These are PRONOUNS, not titles or honorifics
GENDER_PRONOUNS = {
    'korean': {
        'male': ['그는', '그가', '그를', '그의', '그에게', '그도', '그만', '그조차', '그한테'],
        'female': ['그녀는', '그녀가', '그녀를', '그녀의', '그녀에게', '그녀도', '그녀만', '그녀조차', '그녀한테'],
    },
    'japanese': {
        'male': ['彼は', '彼が', '彼を', '彼の', '彼に', '彼も', '彼だけ', '彼こそ'],
        'female': ['彼女は', '彼女が', '彼女を', '彼女の', '彼女に', '彼女も', '彼女だけ', '彼女こそ'],
    },
    'chinese': {
        'male': ['他', '他的', '他们', '他说', '他是', '他在', '他会', '他想'],
        'female': ['她', '她的', '她们', '她说', '她是', '她在', '她会', '她想'],
    },
    'english': {
        'male': [' he ', ' his ', ' him ', ' himself ', 'He ', 'His ', 'Him '],
        'female': [' she ', ' her ', ' hers ', ' herself ', 'She ', 'Her ', 'Hers '],
    }
}

    # Common words - keeping the same for filtering
COMMON_WORDS = {
    '이', '그', '저', '우리', '너희', '자기', '당신', '여기', '거기', '저기',
    '오늘', '내일', '어제', '지금', '아까', '나중', '먼저', '다음', '마지막',
    '모든', '어떤', '무슨', '이런', '그런', '저런', '같은', '다른', '새로운',
    '하다', '있다', '없다', '되다', '하는', '있는', '없는', '되는',
    '것', '수', '때', '년', '월', '일', '시', '분', '초',
    '은', '는', '이', '가', '을', '를', '에', '의', '와', '과', '도', '만',
    '에서', '으로', '로', '까지', '부터', '에게', '한테', '께', '께서',
    # Expanded Common Nouns (to avoid false positive names)
    '사람', '인간', '친구', '동료', '가족', '부모', '형제', '자매', '아이', '어른',
    '학생', '선생', '학교', '회사', '직장', '직원', '사장', '회장', '대표', '작가',
    '마음', '생각', '기분', '감정', '사랑', '우정', '평화', '행복', '희망',
    '문제', '질문', '대답', '정답', '해결', '방법', '이유', '원인', '결과',
    '시간', '장소', '위치', '방향', '거리', '세계', '세상', '우주', '지구',
    '시작', '출발', '도착', '진행', '과정', '끝', '마지막', '결말', '엔딩',
    '아침', '점심', '저녁', '새벽', '밤', '낮', '하루', '평일', '주말', '휴일',
    '수업', '공부', '숙제', '시험', '성적', '점수', '등수', '합격', '불합격',
    '업무', '회의', '보고', '계획', '목표', '성공', '실패', '도전', '포기',
    '돈', '현금', '카드', '통장', '지갑', '가격', '비용', '요금', '월급',
    '집', '방', '거실', '부엌', '화장실', '침대', '책상', '의자', '컴퓨터',
    '음식', '식사', '밥', '빵', '물', '커피', '술', '고기', '야채', '과일',
    '전자', '후자', '평균', '퇴근', '출근', '휴가', '방학', '졸업', '입학',
    '글', '말', '소리', '노래', '음악', '그림', '사진', '영상', '영화',
    '게임', '방송', '뉴스', '신문', '잡지', '책', '소설', '만화',
    # Expanded Adverbs/Conjunctions/Interjections
    '그리고', '그러나', '하지만', '그런데', '그래서', '그러니까', '게다가', '더구나',
    '따라서', '때문에', '왜냐하면', '비록', '만약', '혹시', '과연', '역시',
    '결국', '드디어', '마침내', '이미', '벌써', '아직', '여전히', '계속',
    '항상', '언제나', '늘', '자주', '가끔', '보통', '대체', '도대체',
    '아주', '매우', '너무', '정말', '진짜', '엄청', '완전', '대박', '헐',
    '가장', '제일', '더', '덜', '좀', '조금', '많이', '다시', '또',
    '함께', '같이', '혼자', '따로', '서로', '모두', '다', '전부',
    '그냥', '단지', '오직', '다만', '반드시', '꼭', '제발', '부디',
    '아', '오', '와', '자', '네', '예', '아니오', '음', '응',
    '물론', '사실', '솔직히', '분명', '확실히', '아마', '어쩌면',
    '빨리', '어서', '천천히', '열심히', '잘', '못', '안',
    '바로', '방금', '금방', '곧', '이따가', '나중에',
    '여기저기', '이리저리', '왔다갔다',
    # Common verb/adjective stems or forms often mistaken
    '먹겠네', '가겠네', '하겠네', '좋겠네', '모르겠네',
    '싶어', '하고', '해서', '하면', '하니', '할까', '합시다',
    '좋아', '좋은', '좋게', '싫어', '싫은', '싫게',
    '많아', '많은', '많게', '적어', '적은', '적게',
    '커', '큰', '크게', '작아', '작은', '작게',
    '높아', '높은', '높게', '낮아', '낮은', '낮게',
    '빨라', '빠른', '빠르게', '느려', '느린', '느리게',
    '어려워', '어려운', '쉬워', '쉬운', '쉽게',
    '새로운', '헌', '옛', '첫', '마지막',
    # Internet slang / Chat filler
    'ㅋㅋ', 'ㅎㅎ', 'ㅠㅠ', 'ㅡㅡ', 'ㅇㅇ', 'ㄴㄴ',
    '굿', '굳', '즐', '킨', '감사', '죄송', '축하',
    '안녕', '방가', '하이', '바이',
    '이것', '저것', '그것', '무엇', '아무것',
    '누구', '어디', '언제', '어떻게', '왜', '얼마나'
}
