# 07 — `.tex` and `.pptx` deliverables

This document governs the written deliverables: the meeting deck (`.pptx`), the plan document
(`.tex`) it is built from, the single theory document under `docs/tex/`, and the figures they
share. Research direction, version lifecycle, evaluation, and the ledger are governed by
`00`–`06` and are not restated here.

Read this document before writing or modifying a `.tex` or a `.pptx`. Rule IDs
(M, F, G, H, I, J, K, A, B, C, D, T) are stable, because feedback arrives by ID.

## 1. Scope

- Applies to: the deck, its plan document, the theory document, and the figures they share.
- Does not apply to: experiment design, version open and close, ledger rows, secured
  artifacts. Those are `06_workflow`.
- Statements and proofs of §9 govern the theory document and any formal content wherever it
  appears, including inside a slide.

## 2. 워크플로 (M)

- **M1. 2단계.** 구성안 `.tex` 작성 → Researcher 승인 → pptx 빌드. 승인 전에 pptx를 만들지
  않는다. 구성·순서·수식·문안은 구성안 라운드에서 확정한다. pptx 단계의 수정 비용이 훨씬 크다.
- **M2. 원본은 최초 빌드 시점에 이동한다.** 구성안은 승인·최초 빌드까지의 작업 원본이며, 빌드
  이후의 수정은 deck 소스에서 한다. 구성안으로 역동기화하지 않고 그 시점의 설계 기록으로
  동결한다. 새 발표가 필요하면 새 구성안을 만든다.
- **M3. 블록 포맷.** 슬라이드당 한 블록.
  `\begin{slideblk}{번호}{부제목}{top note}` + `\ag`(그 슬라이드의 목적 한 줄) + `\ct`(빌드 지시)
  + `\tx`(실제 줄배치 그대로) + display 수식 + 선택적 `\keybox`/`\fg`/`\disc` + `\end{slideblk}`.
  `\ag`·`\ct`·`\disc`는 렌더되지 않는 작성자 메모다. **블록 번호는 물리 슬라이드 번호와 1:1**로
  유지한다. divider도 블록을 차지한다.
- **M4. `\ct`는 빌드 지시를 담는다.** figure 재작 플래그, 레이아웃 지시, placeholder 여부, 자산
  경로, 출처 라벨을 여기 명기한다. 빌드 시점에 즉흥 판단하지 않는다. 어떤 asset이나 기록이
  대체되면 대체된 쪽의 `\ct`도 같은 편집에서 갱신한다.
- **M5. 슬라이드 번호 참조는 `\ct`·`\disc`에만.** 본문에 "(S22 참조)" 류 포인터를 넣지 않는다.
  청중에게 필요한 것은 위치가 아니라 그 수식이므로 근거 수식은 top note로 올린다.
- **M6. 재번호는 전면 원자 연산.** 삽입·병합·분할 시 전 슬라이드 재번호 + 섹션 헤더 범위 +
  메모 내 참조 갱신을 한 스크립트에서 수행하고, 보고에 old→new 매핑표를 포함한다.
- **M7. 구성안 게이트.** XeLaTeX 3-pass, errors 0, Overfull 0, 블록 번호 연속성 assert(1..N).
  통과 후에만 전달·빌드로 넘어간다.
- **M8. top note는 필수 필드.** 세 번째 인자로 작성 시점에 확정한다. 내용 규율은 J7(ii).

## 3. 빌드 구조 (F)

- **F1. 소스로 생성한다.** deck은 `python-pptx` 소스로 만든다. 텍스트 변경은 소스 수정 +
  재빌드이지 pptx 직접 편집이 아니다.
- **F2. 혼합 줄은 통째로 하나의 LaTeX 이미지로 조판한다.** 수식 PNG를 textbox의 공백 위에
  얹지 않는다. textbox는 PowerPoint 자체 metric으로, 이미지는 폰트폭 추정 좌표로 배치되므로
  둘은 정렬되지 않는다. 렌더 후 간격을 측정해 이미지를 shift하는 사후 보정은 해법이 아니다.
- **F3. per-instance 튜닝보다 constructive method.** 사례마다 값을 재어 땜질해야 맞는다면 방식
  자체가 틀린 것이다.
- **F4. 본문 폰트 일치.** LaTeX 텍스트 폰트를 deck 본문 폰트(Carlito)로 두어 이미지 줄과 native
  텍스트가 같아 보이게 한다.
- **F5. 빌드 상수는 고정 자산.** 좌표계 960×540 pt, `XL=48` / `XR=912` / `XI=70`,
  `Y_TOP=106` / `Y_BOT=500`, `SZ_H=24` / `SZ_L=20` / `SZ_B=20` / `SZ_SM=16` / `SZ_XS=14.5` /
  `SZ_NOTE=13`. 크기는 소스 상수로만 관리하고 호출부에 숫자를 흩뿌리지 않는다.
- **F6. 편집·검증 프로토콜.** 빌더 편집은 `str.replace` + `assert count==1` 가드로 한다.
  편집 후 shape bbox로 off-slide(x+w>958 / y+h>512)와 partial overlap(±3 pt, 포함 관계 제외)을
  프로그램 검증한다.
- **F7. 입력을 해석하지 못한 컴포넌트는 대체물을 만들지 않고 실패한다.** parser·builder·renderer가
  주어진 내용을 읽지 못하면 오류를 내며, 다른 것을 조용히 생산하는 fallback을 두지 않고, gate
  카운터를 우회하는 축소·근사 경로도 두지 않는다. gate를 통과하는 성능 저하는 crash보다 나쁘다 —
  gate가 잘못된 산출물을 인증하기 때문이다.
  *예:* 구성안이 4개 계의 동역학을 `tabular`에 적었는데 parser는 display만 읽어 `eqs=[]`가 되었다.
  빌더가 `eqs = [기본값1, 기본값2] + eqs[:2]`로 채우고 `zip`이 짧은 쪽에서 끊겨 2개 계의 동역학이
  경고 없이 사라진 채 게이트를 통과했다. 올바른 동작은 `assert len(eqs) == 4`로 빌드를 멈추는
  것이다. 같은 블록에서 폭 초과 시 축소하며 `SHRINK` 카운터를 올리지 않아 `shrink: 0` 게이트도
  우회했다.
- **F8. 순서 변경은 블록 이동으로 하되 정의→사용을 유지한다.** 이동·삽입 후 정의가 사용보다
  앞서는지 확인하고, 섹션 이동 시 섹션 제목과 top note profile도 함께 맞춘다.
- **F9. 알려진 함정.** (i) heredoc raw string이 backslash로 끝나면 SyntaxError — anchor를 `\`로
  끝나지 않게 잡는다. (ii) 편집 anchor에 개행이 포함되면 매칭이 실패해 전량 롤백된다 — anchor
  선정 전에 원문 개행을 확인한다. (iii) deck LaTeX는 `r'...'` single backslash, 텍스트 유니코드는
  `\u`-escape. (iv) `\hline` 뒤에 공백이 없으면 다음 토큰과 붙어 undefined control sequence가 된다.

## 4. 글꼴·크기 (G)

- **G1. 크기 사다리 — 위계로 키우고 여백으로 키우지 않는다.** 본문 20(`SZ_B`) / 보조 display
  16(`SZ_SM`) / 데이터 display 14.5(`SZ_XS`, 하한) / 강조 헤드라인 24(`SZ_H`) / 그림 라벨 ≥13.
  사이의 임의 크기는 지양한다. 공간이 남는다는 이유로 키우지 않는다. 한 슬라이드 안에서 같은
  위계는 같은 크기여야 한다.
- **G2. 제목은 템플릿이 정한다.** 슬라이드마다 개별로 덮어쓰지 않는다.
- **G3. 16 pt 미만은 그림 내부 라벨과 코너 주석에만.**
- **G4. 글꼴은 템플릿 글꼴(Calibri)을 유지하고**, LaTeX 이미지의 텍스트는 metric이 맞는 Carlito를
  쓴다. 수식 크기는 그 줄의 텍스트 크기를 따른다 — 수식 전용 크기 knob을 두지 않는다.
- **G5. 크기는 소스 상수로만 관리한다.**
- **G6. top-right 상태 notation은 섹션별 profile로 관리한다.** divider와 정의 슬라이드는 blank.
- **G7. 자동 축소는 결함 신호다.** 축소가 발동한 줄은 줄 분할이나 문장 축약으로 해소해 사다리
  크기를 회복한다. 빌더는 축소 발동을 세어 보고하고, 전달 게이트는 축소 0을 요구한다.

## 5. 슬라이드 서술 (H)

- **H1. 쉽고 짧게, 그러나 정의 없이 쓰지 않는다.** 간결함이 정의 생략의 핑계가 될 수 없다.
  정의한 대상은 슬라이드에서도 정의를 유지한 채 쓰고, 정의되지 않은 기호를 결론에만 등장시키지
  않는다(§9 A1의 슬라이드 판).
- **H2. 쉬움과 엄밀함은 상충하지 않는다.** 쉽게 만드는 방법은 정의를 빼는 것이 아니라, 정의를
  유지한 채 핵심을 먼저 말하고 추상적 대상을 구체적 근거로 받치는 것이다.
- **H3. noun-phrase가 아니라 완전한 명제로 쓴다.** 한 줄은 정확한 대상과 scope를 갖춘 주장이지
  느슨한 paraphrase가 아니다. 증명으로 확립할 것을 슬라이드에서 가정하지 않는다.
- **H4. 슬라이드당 takeaway 하나 — 그리고 그 takeaway는 메커니즘까지 도달한다.** 결론 문장은
  "무엇"에 더해 "왜/어떻게"가 한 번은 드러나야 한다. 통찰 없이 결과만 있는 슬라이드는 미완성이다.
- **H5. 직관은 말과 그림을 함께 앞쪽에 배치한다.** 시각 자료를 뒤 섹션에 묻지 말고 개념 확립
  지점으로 당긴다.
- **H6. 형태는 전보체, 내용은 완전한 명제.** 최소 단어, 구 단위, 화살표는 `→, ⇒, ⟺`만.
  모호 동사 금지 — "hold"→"satisfied", "carries"→"supports", "dips"→"is negative". 생략하는 것은
  관사·조동사·군더더기이지 대상·조건이 아니다.
- **H7. 약어·은어.** 약어는 첫 사용에 gloss를 병기한다. 내부 은어·은유는 금지하고 직설로 바꾼다.
  자작 명칭은 만들지 않으며 표준 용어가 있으면 그것을 쓴다.
  *금지 사례:* `walk`/`fold`(자작 비유), `ladder`, `corner`, `legs`, `churn`, `worst-ahead`,
  `blind spot`, `anchor`, `carries`.
- **H8. 수식 규율.** display 안에 한글을 넣지 않는다. 중간 기호를 새로 정의하기보다 full
  expression을 우선한다. 행렬·집합에는 차원을 명시하고, display가 폭을 넘으면 분할한다.
- **H9. 설명 단위는 "주장 + 대응 수식"이다.** 텍스트를 몰아 쓴 뒤 수식을 몰아 놓아 대응이 끊기는
  배치를 금지한다.
- **H10. 표기 위생.** (i) 렌더되는 텍스트에 한글을 쓰지 않는다. (ii) 개수는 숫자로 쓴다
  (four → 4). (iii) 미정의 괄호 주석 금지. (iv) 기호 충돌 금지 — 이미 쓰인 기호를 다른 뜻으로
  재사용하지 않는다.

## 6. 내용 엄밀성 (I)

- **I1. 출처에 충실하다.** 슬라이드의 모든 주장은 theory 문서·`results.md`·`ledger.md`의 검증된
  결과로 추적되며, 슬라이드는 제시·압축할 뿐 새로 만들지 않는다. 수학 진술은 원문과 대조 검증 후
  기재하고, 불일치를 발견하면 즉시 보고·교정한다.
- **I2. scope를 정확히 라벨한다.** exact vs limit, iff vs sufficient-only, 조건이 걸리는 영역을
  슬라이드에서도 표기한다. 과대·과소 주장 모두 금지한다. seed 수·pool·device를 밝힌다.
- **I3. 내부 식별자 격리.** checkpoint digest, pool id, run-id, ledger 행 번호, device 태그, 내부
  scoring 이름을 슬라이드 대면 텍스트와 그림에 굽지 않는다. provenance는 manifest와 `ADOPTED.md`에
  두며, digest는 `06_workflow` §6.1에 따라 추적되는 prose 문서 어디에도 쓰지 않는다. 이미지에 구워진
  캡션도 같은 규약의 대상이라 위반 시 그림을 재생성한다.
- **I4. 인용하는 이론 결과에는 증명이 있어야 한다.** 슬라이드가 인용하는 theorem·proposition·
  corollary는 `docs/tex/`의 이론 문서에 증명과 함께 존재해야 한다(§11 T2). 증명이 없으면 인용하지
  않는다.

## 7. 그림 (J)

- **J1. 라벨 있는, 메시지 있는 그림.** 모든 요소에 라벨을 달고 그림마다 한 줄 takeaway를 두며,
  역할별로 color-code하고 non-data ink를 최소화한다. **동일 이미지를 두 슬라이드에 쓰지 않는다** —
  빌더가 assert로 강제한다. placeholder는 `\ct`에 "빌드 시 교체"로 명기한다.
- **J2. block diagram 스타일 일관.** 둥근 사각형, sum junction은 원에 +/− 라벨, 신호 화살표에
  신호명. 되먹임 경로를 생략해 루프가 열린 것처럼 보이게 하지 않는다.
- **J3. 색 절제, 톤 통일.** 개념도이지 상세 배선도가 아니다.
- **J4. 슬라이드 간 흐름.** divider·recap은 허용하되 내용 반복은 금지한다. 같은 display를 두
  슬라이드 본문에 두지 않는다. 병합은 융합이 아니라 적층이다 — 중복을 지우고 위아래로 배치하며
  사이에 공백으로 경계를 보이고 두 top note를 통합한다.
- **J5. figure는 소스 생성기로 만들고 deck과 `.tex`가 자산을 공유한다.** 새로 그리지 않는다.
- **J6. 수식이 드러내는 것을 prose로 재서술하지 않는다.** prose는 수식에 없는 why·consequence·
  mechanism만 담는다.
- **J7. 구성요소 규율.** (i) 부제목은 "무엇을 하는 슬라이드인가 — 어떻게/왜"의 한 구다. 미사여구·
  단어 나열·내부 용어만으로 된 제목은 금지한다. (ii) top note는 가이드 수식만 담는다 — 메타 설명과
  슬라이드 번호 포인터를 넣지 않고, 본문에 있는 수식은 note에서 제거하며, 한 수식은 한 줄, 최대
  약 5줄, 연속 슬라이드에서 같더라도 반복 게시한다. (iii) 보충 슬라이드는 본편보다 유도를 더
  충실히 담고 숨기지 않는다.
- **J8. 그림 자기설명성 — 질문은 결함 신호다.** 청중이나 Researcher가 "이 요소가 무엇이냐"고
  물으면 답변으로 끝내지 않고 그림을 수정한다. 그림이 스스로 설명되지 않으면 발표 중 구두 보완이
  필요하다는 뜻이고, 그것은 그림의 실패다.

## 8. 게이트·검증 (K)

- **K1. 레이아웃 게이트.** 빌더는 매 빌드에 `slides: N  off-slide: 0  shrink: 0`을 출력하고,
  블록 수 == 슬라이드 수, figure 중복 0, partial overlap 0을 assert한다. 하나라도 실패하면
  다운스트림(변환·전달)으로 넘어가지 않는다.
- **K2. 시각 확인에 정직하다.** 도구가 렌더된 슬라이드를 실제로 보여주지 못하면 그렇다고 말하고
  측정에 의존하며, 하지 않은 시각 확인을 했다고 말하지 않는다. 최종 미적 판단은 Researcher에게
  전달된 PDF 확인을 요청한다.
- **K3. 재현 가능한 번들.** 소스, 템플릿 `.pptx`, figure, 의존성 목록, 정확한 빌드 순서를 담은
  README를 포함한다. 전달 전 캐시 없이 clean-room 빌드가 되는지 검증한다. 번들 안의 모든 경로는
  번들 기준 상대경로여야 한다.
- **K4. 텍스트 레이어 게이트.** pptx→pdf 변환 후 `pdftotext`로 (i) LaTeX 리터럴 누출, (ii) 부제목
  기호 소실·문자 깨짐, (iii) 한글 0, (iv) 개수의 단어 표기 0, (v) H7 금지어 0을 점검한다. 발견 0이
  전달 조건이다.
- **K5. 피드백 라운드 프로토콜.** (i) Researcher가 말한 슬라이드 번호가 어느 블록인지 매핑을 먼저
  확인한다. (ii) "패스"는 무수정을 뜻하지 않는다. (iii) 한 지적은 그 슬라이드에서 끝내지 않고
  동종 전 슬라이드에 파급 적용한다(§10 D4). (iv) 라운드마다 보고에 4요소를 담는다: 반영 내역 /
  재번호 매핑 / 신규 발견·불일치 / 잔여 플래그.
- **K6. 수치는 기록에서 인용하고 재계산하지 않는다.** deck에 오르는 수치는 `ledger.md`,
  `results.md`, eval artifact에서 인용하며 인용 지점에서 pool과 seed 수를 밝힌다
  (`06_workflow` §5). 기록에 없는 값은 산출물에 쓰지 않는다.

## 9. Statement and proof writing (A, B, C)

Applies to the theory document and to any formal content wherever it appears, including a
slide.

### A. Statements and definitions

- **A1. Symbol definitions.** Every symbol is defined at first use with `:=` and its dimension,
  space, or role, inside a `definition` or `assumption` environment. Prose-only definitions are
  not allowed. A definition's opening must make its definiendum immediately identifiable; when
  the defined term is a verb, adjective, or adverb (deployable, feasible, invariant), set it in
  `\textbf{\emph{...}}`.
- **A2. Multi-item definitions and assumptions as line-broken lists.** Any `definition` or
  `assumption` carrying two or more conditions is written as an `itemize`/`enumerate`, one item
  per line, never as a single prose paragraph. Label assumption items explicitly.
- **A3. A statement contains the claim only.** A `theorem`/`lemma`/`proposition` holds its
  hypotheses and its result. Analysis, motivation, and "why" go into the proof or into
  interpretation prose.
- **A4. State the scope in the statement.** Each result names its scope where it is stated:
  global / semi-global / regional, unconditional / IC-conditional, exact / limit, and — for a
  measured claim — the seed basis and pool. The reader should not have to infer scope from the
  proof.
- **A5. Formality.** Every design condition, assumption, lemma, proposition, theorem, corollary,
  and proof is written formally and completely: exact hypotheses, exact objects, and what is
  quantified over. A lone inequality or a vague phrase does not substitute for a complete
  statement — an incomplete statement is treated as an error. Do not use a bare colon to carry
  implication: write `\Rightarrow` or a fully quantified sentence.
- **A6. Multi-result statements as enumerated lists.** A statement asserting several distinct
  results lists them as `enumerate[label=(\alph*)]`, one result per item, never a prose run-on.
- **A7. Do not coin terms or notation unnecessarily.** A new name or symbol costs the reader
  lookup, so keep the set of named objects small. Use the standard term where one exists.
  Introduce a new named term or symbol only when the object recurs and naming it is plainly
  easier to read than writing it out; a set or quantity used once or twice is inlined, not
  named. This does not weaken A1: A1 requires that what is used be defined; A7 requires that
  fewer things need defining.

### B. Proofs

- **B1. Equation-forward.** Proofs are a sequence of displayed steps, not prose.
  - **One display = one claim.** Do not pack distinct equalities or implications into one
    display with `,`, `\qquad`, or chained `\iff`.
  - **Implications on their own line.** `A ⟹ B` splits into `A` as one display and `⟹ B` as the
    next.
  - **Each step is preceded only by a short action label** ("complete the square", "fix `P`",
    "minimize"). Do not connect steps with prose paragraphs. Labels are set in `\textbf` and are
    reserved for long proofs or explicit case splits; in a short proof they add noise and are
    omitted.
  - **Label minimalism.** Use a label only where the step genuinely changes task, never to
    restate the line's content in one word.
  - **Case splits use the `cases` environment.**
  - **Forbidden:** cramming steps into one `align`; dissolving the derivation into prose to cut
    length.
- **B2. Conclusion equations are displayed, not inline.** A step's actual conclusion — a sign
  relation, a bound, the target inequality — is a display on its own line. Inline math is only
  for symbols referenced in passing.
- **B3. Quantifier symbols over prose.** In statements and conditions prefer `\forall, \exists,
  \exists!` to "for all / there exists". A fully quantified statement is easier to read fast,
  because the symbols make the logical skeleton visible at a glance.
- **B4. Display-forward beyond proofs.** In statement bodies, remarks, and intuition prose, the
  relation or conclusion that is the point of the paragraph is raised to a display. Inline keeps
  only symbols referenced in passing.

### C. Scope and claim honesty

- **C1. Accurate labels.** Distinguish exact from limit or approximation; mark every bound
  conservative, tight, or over-approximating; state iff versus sufficient-only precisely.
  Neither over-claim nor under-claim. When a bound holds only on a sub-region, state the region
  and do not present it as global.
- **C2. Dependency map.** If a result uses only a subset of the standing assumptions, say which
  ones, so every guarantee is traceable to its hypotheses.
- **C3. State exogenous-signal assumptions explicitly.** Where a result relies on an exogenous
  input, state the assumption it uses. If a result is independent of one, say so.
- **C4. One-directional dependency for mutually-referencing results.** When two results
  reference each other, break the loop in the text with an a priori constant set defined from
  plant data, not from the other result, and state the ordering explicitly, so no proof silently
  assumes the other's conclusion.

## 10. Structure and readability (D)

- **D1. Causal ordering.** Definition then use is one-directional; no forward reference. A new
  result is inserted at its dependency position, not appended at the end.
- **D2. No redundancy.** State each thing once and cross-reference instead of re-deriving.
  A sentence that restates what the adjacent display directly says is redundant with it and is
  deleted; prose next to a display carries the why or the consequence, never a re-reading of the
  symbols. A computation appearing in two proofs is pulled into one lemma and cited twice.
- **D3. concise는 재구조화이지 축약이 아니다.** "줄여라"는 분량 삭감이 아니라 (i) 중복 제거,
  (ii) 이해하기 쉬운 재구조화, (iii) 핵심 수식을 display로 올리고 줄바꿈으로 논리 단위를 분리하는
  것이다. 길이와 명료성이 충돌하면 명료성을 택한다. 이 원칙은 statement 본문·remark·직관 prose·
  채팅 설명 전부에 적용한다.
- **D4. 지시는 그 종류에 적용한다.** 특정 슬라이드나 구절에 대한 수정 지시는 같은 원칙이 걸리는
  다른 자리에도 함께 적용한다. 하나의 수정은 한 종류의 글쓰기에 대한 수정이므로, local로 땜질하고
  일반화하지 않으면 다음 자리에서 같은 지적이 재발한다. 일반화 범위가 갈리면 먼저 후보를 제시하고
  확인받는다.
- **D5. 완료 보고 전 self-audit.** 빌드 게이트 통과는 필요조건일 뿐 완료가 아니다. 새로 쓰거나
  고친 모든 블록을 §5·§6·§9의 기준에 하나씩 대조한 뒤 완료로 보고한다. Researcher가 보는 것은
  게이트가 아니라 서술 품질이다.
- **D6. 품질 지적은 전수 확인으로 마감한다.** 특정 lemma 하나를 고쳤으면 나머지 모든 statement를
  같은 기준으로 점검한다. "이 부분을 고쳤다"가 아니라 "이 종류의 문제가 문서에서 0이다"가 완료
  조건이다.

## 11. 이론 문서 (T)

- **T1. 이론 문서는 `docs/tex/` 아래 하나다.** 버전별·주제별로 분기시키지 않는다. 증명·정리·
  remark의 단일 출처이며, deck과 `results.md`가 인용하는 대상이다.
- **T2. 진술에는 증명이 따른다.** theorem·proposition·corollary·lemma는 증명과 함께 기재한다.
  증명이 아직 없는 주장은 정리 환경에 넣지 않고 open problem으로 remark에 둔다. 형식 요건은 §9.
- **T3. 매 close마다 최신본을 유지한다.** 버전이 새 결과·반증·철회를 만들면 close 전에 이론 문서에
  반영한다. 문서가 뒤처진 채로 close하지 않는다.
- **T3.1. 배포 구현에 대해 진술하는 명제는 전제의 일치를 명시한다.** 명제가 가정하는 상수·형태·
  Lipschitz 상수가 배포된 코드의 그것과 같은지 확인하고, 다르면 그 자리에 기록한다. 배포 상수가
  일반화되면 그것을 인용하는 명제 본문을 함께 고치며, 부속 remark의 수정으로 대신하지 않는다.
- **T4. 인용은 양방향으로 확인한다.** deck과 `results.md`가 이론 문서의 결과를 인용할 때 그 결과가
  현재 문서에 존재하고 증명이 붙어 있는지 확인한다. 반대로 이론 문서가 수치를 인용할 때 그 수치의
  출처 기록과 대조한다.
- **T5. 언어와 렌더링.** theorem·definition·assumption 등 formal 환경의 진술은 영어로, proof와
  remark의 본문 prose는 한국어로 쓴다(한다체, 존댓말·반말 모두 쓰지 않는다). 기술 용어는 영어를
  유지하고 한자·이모지를 쓰지 않는다. 렌더는 XeLaTeX 3-pass에 `fontspec` + Noto Serif CJK KR로
  하고, 전달본은 그 줄을 `\usepackage{kotex}`로 치환한다. 검증은 undefined reference 0, 중복 label
  0, missing character 0이며, 한글이 실제로 렌더되는지 `pdftotext`로 확인한다.
