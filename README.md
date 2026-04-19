## Nhóm thực hiện

**Nhóm 8**
- Trần Đình Nam - **202400063**
- Nguyễn Đức Khải - **202400049**
- Phạm Việt Tiến - **202416366**
  
# Maze Solving with Reinforcement Learning

Dự án này tập trung vào việc giải bài toán mê cung bằng **Reinforcement Learning (RL)**. Mục tiêu là xây dựng một tác tử có khả năng tự học cách di chuyển từ vị trí bắt đầu đến đích thông qua quá trình tương tác với môi trường, thay vì sử dụng các thuật toán tìm đường truyền thống.

## Nội dung chính

Dự án áp dụng các kỹ thuật quan trọng trong học tăng cường, bao gồm:

- **Actor-Critic**: kết hợp mô hình chọn hành động và mô hình đánh giá hành động.
- **Soft Actor-Critic (SAC)**: tối ưu chính sách đồng thời duy trì khả năng khám phá.
- **Curriculum Learning**: huấn luyện tác tử từ mê cung dễ đến khó.
- **Intrinsic Curiosity**: bổ sung động lực nội tại để tăng khả năng khám phá trong môi trường có phần thưởng thưa.

## Mục tiêu dự án

- Mô hình hóa mê cung thành môi trường Reinforcement Learning.
- Huấn luyện tác tử học cách tìm đường tới đích.
- Cải thiện hiệu quả học bằng Curriculum Learning và Intrinsic Curiosity.
- Đánh giá mô hình qua tỉ lệ thành công, số bước di chuyển và tốc độ hội tụ.

## Kết quả mong đợi

Sau khi huấn luyện, tác tử có thể:
- tìm được đường thoát khỏi mê cung,
- thích nghi với nhiều mức độ mê cung khác nhau,
- học hiệu quả hơn so với cách tiếp cận cơ bản nhờ các kỹ thuật hỗ trợ.

## Công nghệ sử dụng

- Python
- PyTorch
- NumPy
- Matplotlib
- Gym-compatible environment / MuJoCo

## Ý nghĩa

Dự án là một ví dụ thực hành về Reinforcement Learning trong bài toán điều hướng, đồng thời có thể mở rộng cho các ứng dụng như robot tự hành, game AI và lập kế hoạch đường đi.
