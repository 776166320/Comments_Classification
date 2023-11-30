import torch
import torch.nn as nn
from transformers import BertModel

class Bert_BiLSTM(nn.Module):
    def __init__(self,
                 lstm_hidden_size: int = 512,
                 lstm_num_layers: int = 1,
                 num_classes: int = 2):
        super(Bert_BiLSTM, self).__init__()

        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.bert = BertModel.from_pretrained("bert-base-chinese")
        self.BiLSTM = nn.LSTM(input_size=768,
                              hidden_size=self.lstm_hidden_size,
                              num_layers=self.lstm_num_layers,
                              batch_first=True,
                              bidirectional=True)
        self.fc = nn.Linear(self.lstm_hidden_size * 2, num_classes)

    def forward(self, inputs):
        """
        Args:
            inputs: {"input_ids": torch.tensor,
                     "token_type_ids": torch.tensor,
                     "attention_mask": torch.tensor}
        Returns:
            model_output: shape=[batch_size, num_classes]
        """
        batch_size = inputs["input_ids"].shape[0]

        # 把输入通过bert
        bert_output = self.bert(**inputs)
        # 把输入通过bilstm，通过nn.lstm样例查看
        lstm_output, (hidden_last, cn_last) = self.BiLSTM(bert_output.last_hidden_state)

        # hidden_last 的维度： [num_directions * num_layers, batch_size, f_dim]
        # cn_last 的维度： [num_directions * num_layers, batch_size, f_dim]

        # 把bilstm的输出整合用fc
        model_output = self.fc(torch.cat([hidden_last[0], hidden_last[1]], dim=1))

        return model_output



if __name__ == '__main__':

    from transformers import BertTokenizer
    model = Bert_BiLSTM()
    tokenizers = BertTokenizer.from_pretrained("bert-base-chinese")

    with torch.no_grad():
        inputs = tokenizers("这是一个测试用例")
        inputs["input_ids"] = torch.tensor(inputs["input_ids"]).unsqueeze(0)
        inputs["token_type_ids"] = torch.tensor(inputs["token_type_ids"]).unsqueeze(0)
        inputs["attention_mask"] = torch.tensor(inputs["attention_mask"]).unsqueeze(0)
        output = model(inputs)
        print(output.shape)


