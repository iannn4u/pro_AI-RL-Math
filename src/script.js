const questionTextElement = document.getElementById("question-text");
const answerInputElement = document.getElementById("answer-input");
const submitButtonElement = document.getElementById("submit-button");
const feedbackTextElement = document.getElementById("feedback-text");
const currentLevelDisplayElement = document.getElementById(
  "current-level-display"
);
const scoreDisplayElement = document.getElementById("score-display");

let currentQuestion = {};
let currentLevel = 1;
let score = 0;
let correctAnswers = 0;
let wrongAnswers = 0;
let q_table = {};
const learning_rate = 0.1;
const discount_factor = 0.9;
let epsilon = 1.0;
const epsilon_decay_rate = 0.995;
const min_epsilon = 0.05;
const maxLevel = 3;
const actions = [-1, 0, 1];

function generateQuestion(levelInput) {
  // Ganti nama parameter agar tidak bentrok dengan global
  let num1, num2, questionString, correctAnswer;
  let questionLevel = levelInput; // Level soal yang akan dihasilkan

  if (levelInput === 1) {
    num1 = Math.floor(Math.random() * 10) + 1;
    num2 = Math.floor(Math.random() * 10) + 1;
    questionString = `${num1} + ${num2} = ?`;
    correctAnswer = num1 + num2;
  } else if (levelInput === 2) {
    num1 = Math.floor(Math.random() * 10) + 5;
    num2 = Math.floor(Math.random() * (num1 - 1)) + 1;
    questionString = `${num1} - ${num2} = ?`;
    correctAnswer = num1 - num2;
  } else {
    // Anggap levelInput >= 3 (atau default jika tidak 1 atau 2)
    num1 = Math.floor(Math.random() * 5) + 1;
    num2 = Math.floor(Math.random() * 5) + 1;
    questionString = `${num1} x ${num2} = ?`;
    correctAnswer = num1 * num2;
    questionLevel = 3; // Soal ini adalah tipe level 3, meskipun inputnya mungkin lebih tinggi
    // Ini penting jika maxLevel Anda adalah 3
  }
  // Pastikan questionLevel yang dikembalikan sesuai dengan soal yang dibuat
  // dan tidak melebihi maxLevel yang bisa ditangani deskripsinya, atau biarkan apa adanya
  // dan biarkan clamping level terjadi di checkAnswer.
  // Untuk konsistensi, jika levelInput adalah 3, maka questionLevel juga 3.
  if (levelInput === 3 && questionLevel !== 3) questionLevel = 3;

  return { text: questionString, answer: correctAnswer, level: questionLevel };
}

function displayQuestion() {
  currentQuestion = generateQuestion(currentLevel);
  questionTextElement.textContent = currentQuestion.text;

  let levelDescription = "";
  if (currentQuestion.level === 1)
    levelDescription = "Pertambahan Mudah"; // Gunakan currentQuestion.level
  else if (currentQuestion.level === 2) levelDescription = "Pengurangan Mudah";
  else if (currentQuestion.level === 3) levelDescription = "Perkalian Mudah";
  else levelDescription = `Level ${currentQuestion.level}`; // Jika ada level lain

  // Pilih span di dalam #current-level-display atau set innerHTML dari #current-level-display
  // Asumsi #current-level-display adalah <p> dan di dalamnya ada <span>
  // Jika #current-level-display adalah elemen yang ingin diisi langsung:
  const levelDisplayTarget = document.querySelector(
    "#current-level-display span"
  );
  if (levelDisplayTarget) {
    levelDisplayTarget.textContent = `${currentQuestion.level} (${levelDescription})`;
  } else {
    // Fallback jika span tidak ditemukan, mungkin #current-level-display itu sendiri targetnya
    currentLevelDisplayElement.textContent = `Level: ${currentQuestion.level} (${levelDescription})`;
  }

  answerInputElement.value = "";
  answerInputElement.focus();
  feedbackTextElement.textContent = "";
  feedbackTextElement.className = "text-lg font-semibold text-slate-700"; // Reset class feedback
}

function checkAnswer() {
  const userAnswerText = answerInputElement.value;
  if (userAnswerText === "") {
    feedbackTextElement.textContent = "Jawaban tidak boleh kosong!";
    feedbackTextElement.className = "text-lg font-semibold text-orange-500"; // Warna oranye untuk warning
    answerInputElement.focus(); // Kembalikan fokus ke input
    return; // Hentikan eksekusi lebih lanjut
  }

  const userAnswer = parseInt(userAnswerText);

  // Handle input bukan angka (NaN)
  if (isNaN(userAnswer)) {
    feedbackTextElement.textContent =
      "Masukkan jawaban berupa angka yang valid!";
    feedbackTextElement.className = "text-lg font-semibold text-orange-500";
    answerInputElement.value = ""; // Kosongkan input yang salah
    answerInputElement.focus();
    return; // Hentikan eksekusi lebih lanjut
  }

  // --- Integrasi RL ---
  const previous_state = currentLevel; // State sebelum menjawab
  let reward = 0;

  if (userAnswer === currentQuestion.answer) {
    feedbackTextElement.textContent = "Benar! 👍";
    feedbackTextElement.className = "text-lg font-semibold text-green-600";
    score += 10;
    correctAnswers++;
    reward = 1; // Reward positif untuk jawaban benar
  } else {
    feedbackTextElement.textContent = `Salah. Jawaban: ${currentQuestion.answer}. 😥`;
    feedbackTextElement.className = "text-lg font-semibold text-red-600";
    wrongAnswers++;
    reward = -1; // Reward negatif untuk jawaban salah
    // Anda bisa membuat reward lebih negatif jika pengguna salah di level mudah,
    // atau kurang negatif jika salah di level sulit.
  }

  // 1. AI memilih tindakan (perubahan level) untuk soal BERIKUTNYA berdasarkan state SAAT INI (previous_state)
  //    Namun, Q-Learning biasanya mengupdate Q(s,a) dan kemudian pindah ke s'.
  //    Jadi, kita tentukan dulu next_level, lalu update Q-Table.

  // Tentukan next_level berdasarkan action yang dipilih AI
  // Action yang dipilih seharusnya berdasarkan `previous_state` sebelum reward diterima.
  // Tapi untuk update Q-table, kita perlu state, action, reward, dan next_state.
  // Mari kita pikirkan alurnya:
  // S_current -> User menjawab -> Dapat Reward -> Tentukan A_next (dari S_current) -> Tentukan S_next
  // Update Q(S_current, A_taken_to_get_reward, Reward, S_next) --- ini sedikit berbeda

  // Pendekatan standar:
  // 1. User ada di `current_state` (misal `previous_state`).
  // 2. User menjawab soal di `current_state`.
  // 3. Dapatkan `reward`.
  // 4. AI memilih `action_to_take_now` berdasarkan `current_state` untuk menentukan `next_real_state`.
  // 5. `updateQTable(current_state, action_that_led_to_current_state, reward, next_real_state)` -- ini masih kurang tepat.

  // Koreksi Alur untuk Q-Learning Sederhana:
  // Saat di state S:
  //  - Pilih A (misal, jenis soal/level untuk dikerjakan SEKARANG). (Kita sudah punya soal dari displayQuestion)
  //  - Lakukan A, dapatkan R dan S' (next_state).
  //  - Update Q(S, A, R, S').
  //  - S = S'.

  // Dalam kasus kita, `currentLevel` adalah state.
  // `action` yang diambil adalah implisit mengerjakan soal di `currentLevel`.
  // Yang kita butuhkan adalah AI memutuskan *perubahan* level.

  // Mari sederhanakan:
  // - `previous_state` adalah `currentLevel` sebelum soal ini dijawab.
  // - `reward` didapat dari jawaban soal di `previous_state`.
  // - AI sekarang harus memilih `action_for_next_level_change` dari `previous_state`.
  const action_level_change = chooseAction(previous_state); // Misal, -1, 0, atau 1

  // Tentukan `next_actual_level`
  let next_actual_level = previous_state + action_level_change;
  // Batasi level agar tidak keluar dari rentang yang valid (misal 1 sampai maxLevel)
  if (next_actual_level < 1) next_actual_level = 1;
  if (next_actual_level > maxLevel) next_actual_level = maxLevel;

  // Sekarang kita punya: previous_state, action_level_change, reward, dan next_actual_level
  // Kita perlu action yang *menyebabkan* kita mendapatkan reward ini.
  // Dalam kasus ini, actionnya adalah 'mengerjakan soal di previous_state'.
  // Q-table kita adalah Q[state][action_untuk_pindah_level].

  // Mari kita re-struktur: Q-table akan belajar nilai dari mengambil action (ubah level) dari sebuah state.
  // State: currentLevel
  // Action: [-1, 0, 1] (turun, tetap, naik)
  // Ketika user menjawab soal di `currentLevel` (state `s`):
  // 1. Dapatkan `reward`.
  // 2. Pilih `action_level_change` (action `a`) dari state `s` menggunakan `chooseAction(s)`.
  // 3. Tentukan `next_level` (state `s'`) = `s + action_level_change`. Pastikan valid.
  // 4. Update Q-table: `updateQTable(s, a, reward, s')`.
  // 5. Set `currentLevel = next_level` untuk soal berikutnya.

  updateQTable(previous_state, action_level_change, reward, next_actual_level);
  currentLevel = next_actual_level; // Update level untuk soal berikutnya

  // --- Akhir Integrasi RL ---

  updateScoreDisplay();
  setTimeout(displayQuestion, 1500);
}

function updateScoreDisplay() {
  const scoreElement = document.querySelector(
    "#score-display span:nth-child(1)"
  );
  const correctElement = document.querySelector(
    "#score-display span:nth-child(2)"
  );
  const wrongElement = document.querySelector(
    "#score-display span:nth-child(3)"
  );

  if (scoreElement) scoreElement.textContent = score;
  if (correctElement) correctElement.textContent = correctAnswers;
  if (wrongElement) wrongElement.textContent = wrongAnswers;
}

submitButtonElement.addEventListener("click", checkAnswer);
answerInputElement.addEventListener("keypress", function (event) {
  if (event.key === "Enter") {
    event.preventDefault();
    checkAnswer();
  }
});

function getQValue(state, action) {
  if (!q_table[state]) {
    q_table[state] = {};
  }
  if (!q_table[state][action]) {
    q_table[state][action] = 0.0;
  }
  return q_table[state][action];
}

function chooseAction(state) {
  let chosen_action;
  if (Math.random() < epsilon) {
    chosen_action = actions[Math.floor(Math.random() * actions.length)];
  } else {
    let best_q_value = -Infinity;
    chosen_action = actions[1];

    if (!q_table[state]) {
      q_table[state] = {};
      for (const action of actions) {
        q_table[state][action] = 0.0;
      }
    }

    const shuffledActions = [...actions].sort(() => Math.random() - 0.5);

    for (const action of shuffledActions) {
      const q_value = getQValue(state, action);
      if (q_value > best_q_value) {
        best_q_value = q_value;
        chosen_action = action;
      }
    }
  }
  if (epsilon > min_epsilon) {
    epsilon *= epsilon_decay_rate;
  }
  return chosen_action;
}

// Tambahkan fungsi ini ke dalam script Anda
function updateQTable(state, action, reward, next_state) {
  // Dapatkan Q-value saat ini untuk state dan action yang diambil
  const old_q_value = getQValue(state, action);

  // Dapatkan Q-value maksimum untuk next_state (nilai dari action terbaik di next_state)
  let max_q_next_state = -Infinity;
  if (q_table[next_state]) {
    // Iterasi semua kemungkinan action di next_state untuk menemukan Q-value tertinggi
    const shuffledNextStateActions = [...actions].sort(
      () => Math.random() - 0.5
    ); // Acak untuk tie-breaking
    for (const next_possible_action of shuffledNextStateActions) {
      max_q_next_state = Math.max(
        max_q_next_state,
        getQValue(next_state, next_possible_action)
      );
    }
  } else {
    // Jika next_state belum pernah dikunjungi (tidak ada di q_table),
    // maka tidak ada pengetahuan tentang Q-value masa depannya, anggap 0.
    max_q_next_state = 0.0;
  }

  // Rumus Q-Learning:
  // Q(s,a) = Q(s,a) + alpha * (reward + gamma * max_Q(s') - Q(s,a))
  const new_q_value =
    old_q_value +
    learning_rate * (reward + discount_factor * max_q_next_state - old_q_value);

  // Pastikan q_table[state] sudah ada
  if (!q_table[state]) {
    q_table[state] = {};
  }
  // Update Q-Table dengan nilai baru
  q_table[state][action] = new_q_value;

  // console.log(`Q-Table Update: S:${state}, A:${action}, R:${reward}, S':${next_state} -> New Q: ${new_q_value.toFixed(3)}`);
  // console.log(JSON.parse(JSON.stringify(q_table))); // Untuk debugging, deep copy agar tidak circular
}

function startGame() {
  q_table = {}; // Kosongkan Q-Table jika ingin reset setiap mulai
  epsilon = 1.0; // Reset epsilon
  currentLevel = 1; // Mulai dari level 1
  score = 0;
  correctAnswers = 0;
  wrongAnswers = 0;
  updateScoreDisplay();
  displayQuestion();
}

startGame();
