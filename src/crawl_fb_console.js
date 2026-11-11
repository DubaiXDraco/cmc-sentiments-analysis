// Dán trực tiếp vào F12 Console trên trang bài viết/Group Facebook công khai
(async function scrapeCMCComments() {
    console.log("Bắt đầu bung toàn bộ bình luận...");
    const sleep = ms => new Promise(r => setTimeout(r, ms));
    
    for (let i = 0; i < 30; i++) {
        let buttons = Array.from(document.querySelectorAll('div[role="button"], span[dir="auto"]'))
            .filter(el => /Xem thêm bình luận|Xem các bình luận trước|View more comments|phản hồi/i.test(el.innerText));
        if (buttons.length === 0) break;
        buttons.forEach(btn => { try { btn.click(); } catch(e){} });
        window.scrollTo(0, document.body.scrollHeight);
        await sleep(1800);
        console.log("Đã mở rộng đợt " + (i + 1) + "...");
    }

    let textElements = document.querySelectorAll('div[dir="auto"][style*="text-align"]');
    let comments = new Set();
    textElements.forEach(el => {
        let text = el.innerText.trim().replace(/\n+/g, ' ');
        if (text.length > 5 && !/^(Thích|Phản hồi|Chia sẻ|Đã chỉnh sửa)$/i.test(text)) {
            comments.add(text);
        }
    });

    let csvContent = "id,raw_text,timestamp\n";
    let idx = 1;
    let now = new Date().toISOString().split('T')[0];
    comments.forEach(cmt => {
        let escaped = '"' + cmt.replace(/"/g, '""') + '"';
        csvContent += idx++ + "," + escaped + "," + now + "\n";
    });

    let blob = new Blob(["\uFEFF" + csvContent], { type: 'text/csv;charset=utf-8;' });
    let link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = "cmc_fb_raw.csv";
    link.click();
    console.log("Đã tải thành công " + comments.size + " mẫu dữ liệu!");
})();
