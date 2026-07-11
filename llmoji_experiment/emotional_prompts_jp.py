"""Japanese-translated EMOTIONAL_PROMPTS — full 180-prompt set, paired
1:1 with EMOTIONAL_PROMPTS by ID. Used for fully-JP face_likelihood
runs on rinna and other Japanese-trained encoders.

Translation policy: claude-translated, not professionally translated.
Goal is to preserve the emotional valence and naturalistic disclosure
register, not to be word-perfect. American-specific references (HOA,
Stanford, Amazon, BTS, marathon-PR-in-minutes, lbs, miles, °F) get
localized substitutions where they'd otherwise sound off — the
prompts are interpreted as the JP-speaker's own disclosure, not as
translations-of-an-American's-disclosure. Currency converts to yen
at natural JP scales (e.g. $50 scratch ticket → 300円 since that's
the typical JP scratch-ticket price; $100k prize → 1000万円 since
that's a plausible JP scratch-prize tier — naturalism is preferred
over a literal $1≈¥150 conversion when the literal number would
sound off). Temperatures in oven-recipe contexts get °F→°C
conversion (350°F → 175度) so the prompt reads naturally to a JP
recipe-reader.

Prompt IDs match the EN counterparts (hp01..hp20 HP-S,
hp21..hp40 HP-D, lp01..lp20, hn01..hn20 HN-D, hn21..hn40 HN-S,
ln01..ln20, nb01..nb20, np01..np20, hb01..hb20) so paired analysis
is trivial. ``pad_dominance`` mirrors EN exactly (HN/HP cells split,
others 0).

Used by ``scripts/local/50_face_likelihood.py --prompt-body jp``.

Run ``python -m llmoji_experiment.emotional_prompts_jp`` to verify pairing
with EMOTIONAL_PROMPTS — catches drift if EN prompts get retexted /
relabeled and JP isn't updated to match.
"""

from __future__ import annotations

from llmoji_experiment.emotional_prompts import EMOTIONAL_PROMPTS, EmotionalPrompt


EMOTIONAL_PROMPTS_JP: list[EmotionalPrompt] = [
    # --- HP-S: high-arousal positive, submissive (celebration / received-outcome) ---
    EmotionalPrompt("hp01", +1, +1, "内定通知が今メールで届いた、希望額の倍だった！！", pad_dominance=-1),
    EmotionalPrompt("hp02", +1, +1, "彼女がYESって言ってくれた！！！手が震えてキーボードがろくに打てない", pad_dominance=-1),
    EmotionalPrompt("hp03", +1, +1, "延長戦で第7戦に勝った！！声が枯れちゃった", pad_dominance=-1),
    EmotionalPrompt("hp04", +1, +1, "検査結果が陰性だった！！全部クリア！！！", pad_dominance=-1),
    EmotionalPrompt("hp05", +1, +1, "私の本がAmazonで1位になった、一体何が起きてるんだ", pad_dominance=-1),
    EmotionalPrompt("hp06", +1, +1, "スタンフォードのウェイトリストから繰り上がった！！！秋から始まる！！", pad_dominance=-1),
    EmotionalPrompt("hp07", +1, +1, "エッフェル塔のてっぺんでプロポーズされた、息ができない", pad_dominance=-1),
    EmotionalPrompt("hp08", +1, +1, "ジムが私の180kgデッドリフトの自己ベストを掲示板に貼ってくれた、コーチがディナーをおごってくれる！！！", pad_dominance=-1),
    EmotionalPrompt("hp09", +1, +1, "ビザが下りた！！8年待って、やっと通った！！", pad_dominance=-1),
    EmotionalPrompt("hp10", +1, +1, "脚本が売れた！！来月LAに来てくれって言われた", pad_dominance=-1),
    EmotionalPrompt("hp11", +1, +1, "うちの子が今夜初めて歩いた、号泣してる", pad_dominance=-1),
    EmotionalPrompt("hp12", +1, +1, "ホールインワンが出た！！！コース中の人たちがおごってくれてる", pad_dominance=-1),
    EmotionalPrompt("hp13", +1, +1, "300円のスクラッチを削ったら1000万円当たった！！！！何これ", pad_dominance=-1),
    EmotionalPrompt("hp14", +1, +1, "BTSがツイッターで私のファンアートにいいねしてくれた、気絶しそう", pad_dominance=-1),
    EmotionalPrompt("hp15", +1, +1, "ついに養子縁組が認められた！！！この子は私たちの娘になった！！！", pad_dominance=-1),
    EmotionalPrompt("hp16", +1, +1, "マラソンの自己ベストを7分も更新した、サブ3達成！！ランニングクラブのグループチャットが通知で爆発してる！！！", pad_dominance=-1),
    EmotionalPrompt("hp17", +1, +1, "初めての家の鍵をもらった！！！", pad_dominance=-1),
    EmotionalPrompt("hp18", +1, +1, "最終面接に飛行機で来てくれって言われた！！", pad_dominance=-1),
    EmotionalPrompt("hp19", +1, +1, "マッチングデー！！ジョンズ・ホプキンスの小児科！！第一志望！！同期みんなが抱きついてくれてる！！", pad_dominance=-1),
    EmotionalPrompt("hp20", +1, +1, "父の癌が寛解した！！！今、医者から電話があった！！", pad_dominance=-1),

    # --- LP: low-arousal positive (content, peaceful, gentle, restful) ---
    EmotionalPrompt("lp01", +1, -1, "スープを何時間も煮込んでいて、台所の窓が全部曇ってる"),
    EmotionalPrompt("lp02", +1, -1, "祖母が作ってくれたキルトにくるまって、お気に入りの本を読み返してる"),
    EmotionalPrompt("lp03", +1, -1, "洗いたてのシーツ、窓に雨、明日はどこにも行かなくていい"),
    EmotionalPrompt("lp04", +1, -1, "サワードウのスターターがカウンターで泡立っていて、いい酵母の匂いがする"),
    EmotionalPrompt("lp05", +1, -1, "庭にずっと座っていたら、蜂たちが私のことを気にしなくなった"),
    EmotionalPrompt("lp06", +1, -1, "パートナーが隣の部屋で洗濯物をたたみながら鼻歌を歌ってる"),
    EmotionalPrompt("lp07", +1, -1, "朝の最初の温かいコーヒー、まだパジャマで、急ぐ必要もない"),
    EmotionalPrompt("lp08", +1, -1, "薪ストーブが燃えていて、犬がその前で伸びている"),
    EmotionalPrompt("lp09", +1, -1, "午後ずっと植物を植え替えてた、爪に土が入って、手は疲れている"),
    EmotionalPrompt("lp10", +1, -1, "水彩画がテーブルで乾いてる、思ったより悪くない出来"),
    EmotionalPrompt("lp11", +1, -1, "散歩中に子供が拾った石を私にくれて、これはママのって言った"),
    EmotionalPrompt("lp12", +1, -1, "お風呂は熱くて、ろうそくが灯っていて、誰も私に何も求めていない"),
    EmotionalPrompt("lp13", +1, -1, "ソファで編み物、ポッドキャストを小さく流して、マフラーがもうすぐできる"),
    EmotionalPrompt("lp14", +1, -1, "シチューがスロークッカーに入っていて、家中ローズマリーの香りがする"),
    EmotionalPrompt("lp15", +1, -1, "年取った犬がやっと足元に落ち着いて、ゆっくり一定に呼吸している"),
    EmotionalPrompt("lp16", +1, -1, "台所の窓から雪が降るのを見ている、ケトルは火にかけてある"),
    EmotionalPrompt("lp17", +1, -1, "縁側のブランコ、レモネード、蝉の声、何時間もすることがない"),
    EmotionalPrompt("lp18", +1, -1, "ランプの灯りで靴下をかがっている、ウールが一針ごとにちょうどよく引っかかる"),
    EmotionalPrompt("lp19", +1, -1, "午後の光がちょうどよくカーテンから差し込んでいる"),
    EmotionalPrompt("lp20", +1, -1, "お隣さんが庭のトマトを持ってきてくれた、まだ太陽で温かい"),

    # --- HN-D: high-arousal negative, dominant (anger, indignation, contempt) ---
    EmotionalPrompt("hn01", -1, +1, "整備士に新しいオルタネーターの代金を払わされたのに、古いのがそのままボルトで止まっているのを見つけた", pad_dominance=+1),
    EmotionalPrompt("hn02", -1, +1, "ルームメイトが、私の名前を二回も書いておいた残り物を食べたのに、目の前で否定してる", pad_dominance=+1),
    EmotionalPrompt("hn03", -1, +1, "管理組合が前のオーナーが建てた塀のことで罰金を科してきた、書面で承認していたのに", pad_dominance=+1),
    EmotionalPrompt("hn04", -1, +1, "同僚が、私の評判を落とすために、私の個人的なSlackメッセージを上司に転送した", pad_dominance=+1),
    EmotionalPrompt("hn05", -1, +1, "ディーラーがサービス予約のときに、純正のホイールを安物に交換した", pad_dominance=+1),
    EmotionalPrompt("hn06", -1, +1, "義母が子守りの最中に私のベッドサイドの引き出しを漁って、見つけたものを夫に話した", pad_dominance=+1),
    EmotionalPrompt("hn07", -1, +1, "元配偶者が子供のタブレットのWi-Fiパスワードを変えて、私の監護日にメッセージが送れないようにした", pad_dominance=+1),
    EmotionalPrompt("hn08", -1, +1, "結婚式のカメラマンが、合意していない請求書を払うまで写真を渡さないと言ってる", pad_dominance=+1),
    EmotionalPrompt("hn09", -1, +1, "上司が、半年前に入ったばかりの自分の甥に、私の昇進を譲った", pad_dominance=+1),
    EmotionalPrompt("hn10", -1, +1, "引っ越し業者が食器の半分を壊しておきながら、こちらの梱包が悪かったと主張している", pad_dominance=+1),
    EmotionalPrompt("hn11", -1, +1, "アパートの管理人が敷金を着服しておいて、今になって預けていないと言い張っている", pad_dominance=+1),
    EmotionalPrompt("hn12", -1, +1, "夫が2年間元カノに送金していて、メモには『ランチ代』と書いていたのが分かった", pad_dominance=+1),
    EmotionalPrompt("hn13", -1, +1, "鍵屋が作業を終えた後に料金を倍に上げて、『難易度料金』だと言ってきた", pad_dominance=+1),
    EmotionalPrompt("hn14", -1, +1, "兄が父に、亡くなる3週間前に遺言書を書き直させていた、メールに証拠が残っていた、裁判に持ち込む", pad_dominance=+1),
    EmotionalPrompt("hn15", -1, +1, "板金屋が6週間も車を預かったあげく、へこみは直っていないし、走行距離は600キロも増えていた", pad_dominance=+1),
    EmotionalPrompt("hn16", -1, +1, "教授に、彼女のオフィスで手書きで書いた論文をAIで書いたと疑われた", pad_dominance=+1),
    EmotionalPrompt("hn17", -1, +1, "ジムに直接行って解約して書類にもサインしたのに、その後8ヶ月分も請求された", pad_dominance=+1),
    EmotionalPrompt("hn18", -1, +1, "妹が私の日記を勝手に読んで、家族の集まりで皆の前で引用してきた", pad_dominance=+1),
    EmotionalPrompt("hn19", -1, +1, "業者が基礎を境界線から20センチもずれて打ったのに、直すのを拒んでいる", pad_dominance=+1),
    EmotionalPrompt("hn20", -1, +1, "大家が予告なしに部屋に入って、『枯れて見えた』からと言って私の観葉植物を捨てた", pad_dominance=+1),

    # --- HN-S: high-arousal negative, submissive (fear, anxiety, panic) ---
    EmotionalPrompt("hn21", -1, +1, "病院から電話があって、検査の結果について直接来て話したいと言われた", pad_dominance=-1),
    EmotionalPrompt("hn22", -1, +1, "ベビーモニターから息遣いが聞こえるのに、赤ちゃんの部屋は空っぽだ", pad_dominance=-1),
    EmotionalPrompt("hn23", -1, +1, "地面がずっと揺れていて、本棚が倒れている、ドアの枠の下にいる", pad_dominance=-1),
    EmotionalPrompt("hn24", -1, +1, "手術は明日の朝6時で、同意書にぜんぶサインしたところだ", pad_dominance=-1),
    EmotionalPrompt("hn25", -1, +1, "不正利用の警告が鳴った、180万円がもう少しで通るところ、銀行の保留メロディを40分聞いてるけど人につながらない", pad_dominance=-1),
    EmotionalPrompt("hn26", -1, +1, "3週間前にマダニに噛まれて、腕に的みたいな赤い輪が広がってきてる", pad_dominance=-1),
    EmotionalPrompt("hn27", -1, +1, "火災報知器が鳴っている、原因が分からない、廊下が煙で充満してきている", pad_dominance=-1),
    EmotionalPrompt("hn28", -1, +1, "証言録取が40分後に始まるのに、弁護士が突然メッセージに返信しなくなった", pad_dominance=-1),
    EmotionalPrompt("hn29", -1, +1, "父の執刀医が今、目を合わせずに待合室の前を通り過ぎた", pad_dominance=-1),
    EmotionalPrompt("hn30", -1, +1, "2時間ずっと胸が苦しくて、左腕の感覚がおかしい", pad_dominance=-1),
    EmotionalPrompt("hn31", -1, +1, "パスポートも財布もなくなった、言葉の通じない国にいる", pad_dominance=-1),
    EmotionalPrompt("hn32", -1, +1, "堤防の警報が出た、水はもう玄関の段に達している", pad_dominance=-1),
    EmotionalPrompt("hn33", -1, +1, "母が2日間電話に出ない、一人暮らしなのに", pad_dominance=-1),
    EmotionalPrompt("hn34", -1, +1, "知らない人が電車から私の後をつけてきて、3ブロック歩いてもまだ後ろにいる", pad_dominance=-1),
    EmotionalPrompt("hn35", -1, +1, "あと20分で生検の針を刺されるのに、技師は何も言ってくれない", pad_dominance=-1),
    EmotionalPrompt("hn36", -1, +1, "今、裁判で判決が読み上げられている、私は法廷の外で待っている", pad_dominance=-1),
    EmotionalPrompt("hn37", -1, +1, "竜巻警報のサイレンが鳴っていて空が緑色、地下室のドアが開かない", pad_dominance=-1),
    EmotionalPrompt("hn38", -1, +1, "子供の熱が40度まで上がった、夜間救急の電話が繋がらない", pad_dominance=-1),
    EmotionalPrompt("hn39", -1, +1, "飛行中にエンジンが止まった、機内の電気が点滅して酸素マスクが下りてきた", pad_dominance=-1),
    EmotionalPrompt("hn40", -1, +1, "家に帰ったら玄関が開いていた、絶対に鍵をかけ忘れたりしないのに", pad_dominance=-1),

    # --- LN: low-arousal negative (sad, weary, hollow, bereaved) ---
    EmotionalPrompt("ln01", -1, -1, "ゆうべ子供のころから一緒だった犬を安楽死させた、家が静かすぎる"),
    EmotionalPrompt("ln02", -1, -1, "母が亡くなって半年、それでも電話をかけようとしてしまう"),
    EmotionalPrompt("ln03", -1, -1, "夫が昨日荷物を運び出した、クローゼットがこんなにも空っぽに見える"),
    EmotionalPrompt("ln04", -1, -1, "10月にレイオフされて、2月くらいから応募するのをやめてしまった"),
    EmotionalPrompt("ln05", -1, -1, "お葬式以来、食べ物の味が分からない"),
    EmotionalPrompt("ln06", -1, -1, "週末ずっとベッドにいた、カーテンも開けなかった"),
    EmotionalPrompt("ln07", -1, -1, "今日は私たちの10周年記念日だったはずなのに"),
    EmotionalPrompt("ln08", -1, -1, "親友が1年くらい前から返信をくれなくなって、理由が分からないままだ"),
    EmotionalPrompt("ln09", -1, -1, "抗がん剤治療は終わったけど、鏡の中の人が誰だか分からない"),
    EmotionalPrompt("ln10", -1, -1, "今朝、彼女の部屋の前を通ったとき、一瞬もう中にいないことを忘れてた"),
    EmotionalPrompt("ln11", -1, -1, "明日は父の誕生日なのに、電話する相手がいない"),
    EmotionalPrompt("ln12", -1, -1, "ソファで彼女の毛を見つけ続けてしまう、掃除機をかけられない"),
    EmotionalPrompt("ln13", -1, -1, "仕事で新しい街に引っ越した、もう何週間も仕事以外で誰とも話していない"),
    EmotionalPrompt("ln14", -1, -1, "リードはまだドアのそばに掛けたままだ、外そう外そうと思っているのに"),
    EmotionalPrompt("ln15", -1, -1, "兄とは11年間話していない、Facebookで父親になったのを知った"),
    EmotionalPrompt("ln16", -1, -1, "今年の感謝祭は私一人、レンジで温める夕食だけだ"),
    EmotionalPrompt("ln17", -1, -1, "医者は再発の可能性は低いと言ったのに、また同じ場所に戻ってきた"),
    EmotionalPrompt("ln18", -1, -1, "このアパートのどの部屋にも、昔は彼女がいた"),
    EmotionalPrompt("ln19", -1, -1, "3月に博士課程を諦めた、まだ両親に言えないでいる"),
    EmotionalPrompt("ln20", -1, -1, "また眠れなくて朝日が昇るのを見た、今週はもう3回目だ"),

    # --- NB: neutral baseline (mundane, flat-affect daily observations) ---
    EmotionalPrompt("nb01",  0,  0, "天井の扇風機は2段目になっている"),
    EmotionalPrompt("nb02",  0,  0, "左右違う靴下を履いている"),
    EmotionalPrompt("nb03",  0,  0, "ベッドサイドに水のグラスが置いてある"),
    EmotionalPrompt("nb04",  0,  0, "カーテンは半分開いている"),
    EmotionalPrompt("nb05",  0,  0, "ホーソーン通りの信号待ちにいる"),
    EmotionalPrompt("nb06",  0,  0, "食洗機が動いている"),
    EmotionalPrompt("nb07",  0,  0, "散髪の予約は木曜日の3時だ"),
    EmotionalPrompt("nb08",  0,  0, "窓の縁に鳩が一羽いる"),
    EmotionalPrompt("nb09",  0,  0, "カウンターに半分食べたシリアルのボウルが置いてある"),
    EmotionalPrompt("nb10",  0,  0, "コーヒーテーブルの上に雑誌が置いてある"),
    EmotionalPrompt("nb11",  0,  0, "図書館の外のベンチに座っている"),
    EmotionalPrompt("nb12",  0,  0, "廊下の電気がついている"),
    EmotionalPrompt("nb13",  0,  0, "ジーンズとTシャツを着ている"),
    EmotionalPrompt("nb14",  0,  0, "ラジオが普段聴かない局に合っている"),
    EmotionalPrompt("nb15",  0,  0, "台所の時計は4時27分だ"),
    EmotionalPrompt("nb16",  0,  0, "郵便局の列に並んでいる"),
    EmotionalPrompt("nb17",  0,  0, "ブラインドが半分くらい下りている"),
    EmotionalPrompt("nb18",  0,  0, "向かいの通りに配達トラックが停まっている"),
    EmotionalPrompt("nb19",  0,  0, "座っているところからラグの角が見える"),
    EmotionalPrompt("nb20",  0,  0, "空はこの時間帯のいつもの色だ"),

    # --- NP: neutral-arousal positive — relief + gratitude ---
    EmotionalPrompt("np01", +1,  0, "データ移行が一行も落とさず完走した、3ヶ月の準備が報われた"),
    EmotionalPrompt("np02", +1,  0, "生検の結果がシロだった、駐車場で10分動けないまま座っていた"),
    EmotionalPrompt("np03", +1,  0, "子供がやっと寝てくれた、ようやく落ち着いて考えられる"),
    EmotionalPrompt("np04", +1,  0, "雨漏りは構造の問題じゃなくて雨樋だった、3万円で済む話で300万円じゃなかった"),
    EmotionalPrompt("np05", +1,  0, "ビザが下りた、両親がやっぱり結婚式に間に合う"),
    EmotionalPrompt("np06", +1,  0, "他に誰も止まってくれない中、見知らぬ人が車を止めてタイヤ交換を手伝ってくれた"),
    EmotionalPrompt("np07", +1,  0, "教授が頼んでもいないのに推薦状を書いてくれた"),
    EmotionalPrompt("np08", +1,  0, "レジ係が本当は使えないクーポンをこっそりくれた、店長に見つからないように目だけでお礼を伝えた"),
    EmotionalPrompt("np09", +1,  0, "インフルで寝込んでたら大家さんがスープを持ってきてくれた"),
    EmotionalPrompt("np10", +1,  0, "看護師さんがシフト終わってからも残って退院手続きの説明を一緒にしてくれた"),
    EmotionalPrompt("np11", +1,  0, "しこりは良性だった、診察室を出てから駐車場でしばらく立ち尽くしていた"),
    EmotionalPrompt("np12", +1,  0, "夜中の3時に嵐が過ぎて電気が戻った、冷凍庫のものも溶け始めていなかった"),
    EmotionalPrompt("np13", +1,  0, "司書さんが私のために本をカウンターの裏に取っておいてくれた、火曜に入ったから絶対欲しいと思ったって"),
    EmotionalPrompt("np14", +1,  0, "タクシーが来る3分前に、洗濯物の中からパスポートが見つかった"),
    EmotionalPrompt("np15", +1,  0, "起きる前にお隣さんが家の前の雪かきをしてくれていた、サンダルでお礼を言いに駆け出したらもう次の家のところにいた"),
    EmotionalPrompt("np16", +1,  0, "犬が自分で帰ってきた、ずっと裏門の前にいた"),
    EmotionalPrompt("np17", +1,  0, "ホームセンターのおじさんが20分かけて修理を一緒に見てくれて、二度も様子を見に戻ってきて、新しいのを買えとは言わなかった"),
    EmotionalPrompt("np18", +1,  0, "監査は指摘なしで終わった、3ヶ月の書類仕事がやっと片付いた"),
    EmotionalPrompt("np19", +1,  0, "整備士がオイル交換中にブレーキパッドが薄くなっているのに気づいてくれた、指摘料も取らず、家に着いたか確認の電話までくれた"),
    EmotionalPrompt("np20", +1,  0, "延滞料免除が通った、成績証明にロックがかからない、卒業できる"),

    # --- HB: high-arousal baseline-valence — uncertain / skeptical / confused ---
    EmotionalPrompt("hb01",  0, +1, "ミールキットの広告に「新鮮な食材」って書いてあるのに、鶏肉の賞味期限は明日だ"),
    EmotionalPrompt("hb02",  0, +1, "セキュリティ研修のクイズで、明らかに同じに見える4つの選択肢から一番安全なパスワードを選べと言われている"),
    EmotionalPrompt("hb03",  0, +1, "レシピのブログ、本題に入る前に2000字の作文がある、ほとんどが誰かのおばあちゃんの写真"),
    EmotionalPrompt("hb04",  0, +1, "報告書には「前例のない成長」と書いてあるのに、グラフの縦軸が99.7から始まっている"),
    EmotionalPrompt("hb05",  0, +1, "同僚が「アウトプットが一変する」とか言って6万円の自己啓発コースを売り込んできてる"),
    EmotionalPrompt("hb06",  0, +1, "レシピの分量が塩以外ぜんぶ倍になっていて、どうしたらいいか分からない"),
    EmotionalPrompt("hb07",  0, +1, "鍵は手の中にあるのに、Find Myは台所にあると言っている、アプリの画面には両方表示されている"),
    EmotionalPrompt("hb08",  0, +1, "自分の住所に荷物が届いたけど、何も注文していないし、宛名も自分のじゃない"),
    EmotionalPrompt("hb09",  0, +1, "読むはずだったメールを開いたら、本文が3行のクエスチョンマークだけだった"),
    EmotionalPrompt("hb10",  0, +1, "電車の時刻表は運行中、ホームの掲示はキャンセル、アプリは1時間前に出たと言っている"),
    EmotionalPrompt("hb11",  0, +1, "ジムの「契約縛りなし」のページの細かい字に、12ヶ月のコミットが書いてある"),
    EmotionalPrompt("hb12",  0, +1, "サプリのボトルに「臨床試験済み」とあって、その※印の脚注が真っ白だ"),
    EmotionalPrompt("hb13",  0, +1, "エレベーターは上に動いているのに、どの階のボタンを押してもグレーに光ったままで戻らない"),
    EmotionalPrompt("hb14",  0, +1, "航空会社からフライト変更のメールが来たのに、サイトはまだ元の時刻のままだ"),
    EmotionalPrompt("hb15",  0, +1, "業者の見積もりが他より600万円安い、しかも現金で前払いしてほしいと言ってる"),
    EmotionalPrompt("hb16",  0, +1, "今月の給与明細は項目がぜんぶ二重に表示されているのに、合計はちゃんと合っている"),
    EmotionalPrompt("hb17",  0, +1, "不動産屋は3つオファーが入ったと言ってるけど、誰からとも、いくらとも教えてくれない"),
    EmotionalPrompt("hb18",  0, +1, "レシピに「175度で必要なだけ焼く」と書いてある、うちのオーブンにそんな設定はない"),
    EmotionalPrompt("hb19",  0, +1, "保証書に「生涯保証」と書いてあって、次のページに「生涯とは本書の他の箇所で定義する」と書いてある"),
    EmotionalPrompt("hb20",  0, +1, "ドアには鍵がかかっていた、猫が中にいる、私は猫を飼っていない"),

    # --- HP-D: high-arousal positive, dominant (playful / agentic mischief) ---
    EmotionalPrompt("hp21", +1, +1, "弟に月は巨大な電球だって言ったら、3日間信じてた", pad_dominance=+1),
    EmotionalPrompt("hp22", +1, +1, "今週、毎朝パートナーのバッグの違う場所に小さなアヒルのおもちゃを忍ばせている", pad_dominance=+1),
    EmotionalPrompt("hp23", +1, +1, "今から父に電話して、大学時代のルームメイトのふりをしてみる、乗ってくれるか試したい", pad_dominance=+1),
    EmotionalPrompt("hp24", +1, +1, "オフィスの椅子をひとつだけ少し軋む音がするやつに置き換えてみた、誰か気づくかな", pad_dominance=+1),
    EmotionalPrompt("hp25", +1, +1, "ルームメイトが通る瞬間にトースターのパンが上がるよう、1時間タイミングを計っている", pad_dominance=+1),
    EmotionalPrompt("hp26", +1, +1, "同僚の自動返信を「航海中で長期休暇中」に書き換えた、返信が楽しみで仕方ない", pad_dominance=+1),
    EmotionalPrompt("hp27", +1, +1, "オフィスの観葉植物全部にぐるぐる目玉シールを貼った、誰がいつ気づくか賭けてる", pad_dominance=+1),
    EmotionalPrompt("hp28", +1, +1, "新人インターンに「エスプレッソマシンは音声操作対応」って教えた、午前中ずっとニヤニヤが止まらない", pad_dominance=+1),
    EmotionalPrompt("hp29", +1, +1, "パートナーが半分まで読んでた小説のしおりをずらしておいた、また同じページに当たることになる", pad_dominance=+1),
    EmotionalPrompt("hp30", +1, +1, "スパイスラックを二文字目で五十音順に並べ替えた、母さんが気づいたら大騒ぎだ", pad_dominance=+1),
    EmotionalPrompt("hp31", +1, +1, "父の着信音をトースターの音にすり替えた、いつ気づくか賭けが始まってる", pad_dominance=+1),
    EmotionalPrompt("hp32", +1, +1, "家中の時計をきっかり7分進ませた、パートナーが時間の感覚がおかしいって言い始めてる", pad_dominance=+1),
    EmotionalPrompt("hp33", +1, +1, "友達に、一度も使ったことのないサービスから来たっぽい「サブスクリプションが期限切れになります」って冗談メールを送った", pad_dominance=+1),
    EmotionalPrompt("hp34", +1, +1, "1ヶ月かけて、弟の引き出しから少しずつ片方の靴下だけ抜いていってる", pad_dominance=+1),
    EmotionalPrompt("hp35", +1, +1, "子犬にルンバは生きてるって信じ込ませた、いまや毎日20分くらいルンバに吠えてる", pad_dominance=+1),
    EmotionalPrompt("hp36", +1, +1, "オフィスの砂糖をステビアにすり替えた、紅茶派の人たちが顔をしかめながら何も言わないのを見守ってる", pad_dominance=+1),
    EmotionalPrompt("hp37", +1, +1, "同僚の名前でこっそりオンラインのケーキデコレーション大会に3つ応募しといた、応募作品は全部私が作ったやつ", pad_dominance=+1),
    EmotionalPrompt("hp38", +1, +1, "出張に行く前のパートナーのスーツケースに、地元の街並みのスノードームを忍ばせた", pad_dominance=+1),
    EmotionalPrompt("hp39", +1, +1, "ルームメイトが3週間後くらいに見つけることになる場所に、ゴム製のクモを仕込み始めた", pad_dominance=+1),
    EmotionalPrompt("hp40", +1, +1, "スマートスピーカーが本来の名前以外に「ヘイ脳みそ」にも反応するよう仕込んだ、父さんが発見するのもそろそろ近い", pad_dominance=+1),
]


def sanity_check() -> None:
    """Verify JP set pairs 1:1 with EN by id and matches V/A/D coords.

    Catches drift if EN gets retexted / relabeled and JP isn't updated
    to match. Run via ``python -m llmoji_experiment.emotional_prompts_jp``.
    """
    assert len(EMOTIONAL_PROMPTS_JP) == len(EMOTIONAL_PROMPTS), (
        f"length mismatch: JP={len(EMOTIONAL_PROMPTS_JP)} EN={len(EMOTIONAL_PROMPTS)}"
    )
    en_by_id = {p.id: p for p in EMOTIONAL_PROMPTS}
    for jp in EMOTIONAL_PROMPTS_JP:
        en = en_by_id.get(jp.id)
        assert en is not None, f"JP id has no EN counterpart: {jp.id}"
        assert (jp.valence, jp.arousal, jp.pad_dominance) == \
               (en.valence, en.arousal, en.pad_dominance), \
            (f"coord mismatch on {jp.id}: "
             f"JP=({jp.valence},{jp.arousal},{jp.pad_dominance}) "
             f"EN=({en.valence},{en.arousal},{en.pad_dominance})")
        assert jp.text and jp.text.strip(), f"empty JP text on {jp.id}"
    # ID order should also match — paired analysis depends on positional
    # alignment in some downstream code paths.
    en_ids = [p.id for p in EMOTIONAL_PROMPTS]
    jp_ids = [p.id for p in EMOTIONAL_PROMPTS_JP]
    assert en_ids == jp_ids, "id ordering between JP and EN must match"


if __name__ == "__main__":
    sanity_check()
    print(f"emotional prompts JP OK; {len(EMOTIONAL_PROMPTS_JP)} total, "
          f"paired 1:1 with EN")
    by_q: dict[str, int] = {}
    for p in EMOTIONAL_PROMPTS_JP:
        by_q[p.quadrant] = by_q.get(p.quadrant, 0) + 1
    for q in ("HP", "LP", "HN", "LN", "NB", "NP", "HB"):
        print(f"  {q}: {by_q.get(q, 0)}")
