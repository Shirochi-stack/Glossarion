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
        
            # Classical/formal honorifics
        '공', '옹', '군', '양', '낭', '랑', '생', '자', '부', '모', '시', '제', '족하',
        
            # Royal/noble address forms
        '마마', '마노라', '대감', '영감', '나리', '도령', '낭자', '아씨', '규수',
        '각하', '전하', '폐하', '저하', '합하', '대비', '대왕', '왕자', '공주',
        
            # Buddhist/religious
        '스님', '사부님', '조사님', '큰스님', '화상', '대덕', '대사', '법사',
        '선사', '율사', '보살님', '거사님', '신부님', '목사님', '장로님', '집사님',
        
            # Confucian/scholarly
        '부자', '선생', '대인', '어른', '어르신', '존자', '현자', '군자', '대부',
        '학사', '진사', '문하생', '제자',
        
            # Kinship honorifics
        '어르신', '할아버님', '할머님', '아버님', '어머님', '형님', '누님',
        '아주버님', '아주머님', '삼촌', '이모님', '고모님', '외삼촌', '장인어른',
        '장모님', '시아버님', '시어머님', '처남', '처형', '매형', '손님',
        
            # Verb-based honorific endings and speech levels
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
        r'\b(Tianzi|Huangdi|Huanghou|Taizi|Qinwang|Junwang|Beile|Beizi|Gongzhu|Gege)\b',
        r'\b(Chengxiang|Zaixiang|Taishi|Taifu|Taibao|Taiwei|Situ|Sikong|Dasima)\b',
        r'\b(Shogun|Daimyo|Samurai|Ronin|Ninja|Tenno|Mikado|Kampaku|Sessho)\b',
        r'\b(Taewang|Wangbi|Wanghu|Seja|Daegun|Gun|Ongju|Gongju|Buma)\b'
    ]
}

    # Expanded Chinese numbers including classical forms
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